# teacher_generate.py
# Two-stage pipeline WITH APPEND:
#   A) Generate single questions via parallel calls
#   B) Answer each question with top-k logprobs
#
# Stacking runs:
#   - If questions.arrow / answers.arrow exist, we read them,
#     continue qid from max(existing_qid)+1, and write back old+new.
#
# REQUIREMENTS:
#   pip install omegaconf requests pyarrow "transformers>=4.43.0"
#
# NOTES:
# - vLLM OpenAI-compatible /chat/completions
# - top-k logprobs per generated token (storage-friendly KD)
# - We manually insert [USER]/[AI] in text and tokenize with add_special_tokens=False
#   so role/loss masks line up exactly, independent of post-processors.

import os, sys, json, random
from pathlib import Path
from typing import Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from omegaconf import OmegaConf
import pyarrow as pa
import pyarrow.ipc as pa_ipc
from transformers import AutoTokenizer

DATA_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DATA_DIR.parent


def _load_config(user_path: str | None) -> OmegaConf:
    base = OmegaConf.load(PROJECT_ROOT / "Global_Config.yml")
    default = OmegaConf.load(DATA_DIR / "Config.yml")
    if user_path:
        user_cfg = OmegaConf.load(user_path)
        return OmegaConf.merge(base, default, user_cfg)
    return OmegaConf.merge(base, default)


def _resolve_student_tokenizer(cfg, teacher_cfg) -> Tuple[str, str | None]:
    override_dir = getattr(teacher_cfg, "student_tokenizer_dir", None)
    override_name = getattr(teacher_cfg, "student_tokenizer_name", None)
    tok_cfg = cfg.tokenizer
    base_prefix = Path(cfg.paths.get("data_root", "")) if "paths" in cfg else None

    if override_dir:
        dir_path = Path(override_dir)
        if not dir_path.is_absolute():
            if base_prefix and base_prefix.exists():
                dir_path = (base_prefix / dir_path).resolve()
            else:
                dir_path = (PROJECT_ROOT / override_dir).resolve()
        if dir_path.exists():
            return str(dir_path), None
        return str(dir_path), getattr(tok_cfg, "cache_dir", None)

    if override_name:
        return override_name, getattr(tok_cfg, "cache_dir", None)

    if tok_cfg.use_custom:
        custom_path = Path(tok_cfg.custom_path)
        if not custom_path.is_absolute():
            if base_prefix and base_prefix.exists():
                custom_path = (base_prefix / custom_path).resolve()
            else:
                custom_path = (PROJECT_ROOT / tok_cfg.custom_path).resolve()
        if custom_path.exists():
            return str(custom_path), None
        return tok_cfg.custom_path, getattr(tok_cfg, "cache_dir", None)

    return tok_cfg.name, getattr(tok_cfg, "cache_dir", None)


def _resolve_output_path(base_prefix: Path | None, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if base_prefix is not None:
        return (base_prefix / path).resolve()
    return (PROJECT_ROOT / path).resolve()


# -----------------------
# Small HTTP client bits
# -----------------------
def _jdump(x):
    try:
        import orjson
        return orjson.dumps(x).decode()
    except Exception:
        return json.dumps(x, ensure_ascii=False)

def chat_complete(
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    top_p: float,
    max_tokens: int,
    *,
    # vLLM/OpenAI chat: set logprobs=True and top_logprobs=int
    logprobs: bool | None = None,
    top_logprobs: int = 0,
    prompt_logprobs: int = 0,
    response_format: Dict[str, Any] | None = None,
    timeout: int = 120,
):
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_tokens),
    }

    if top_logprobs and int(top_logprobs) > 0:
        payload["logprobs"] = True if logprobs is None else bool(logprobs)
        payload["top_logprobs"] = int(top_logprobs)
    elif logprobs is True:
        payload["logprobs"] = True

    if prompt_logprobs and int(prompt_logprobs) > 0:
        payload["prompt_logprobs"] = int(prompt_logprobs)

    if response_format:
        payload["response_format"] = response_format

    r = requests.post(url, headers=headers, data=_jdump(payload), timeout=timeout)
    r.raise_for_status()
    return r.json()


# --------------------------------------------------
# Parse vLLM chat logprobs into compact JSON strings
# --------------------------------------------------
def parse_topk_json(choice: Dict[str, Any]) -> List[str]:
    """
    Robustly parse vLLM/OpenAI-style chat logprobs.

    We handle BOTH shapes:
    - dict:   {"top_logprobs": {" tok": -0.1, "The": -0.4, ...}}
    - list:   {"top_logprobs": [{"token": " tok", "logprob": -0.1}, ...]}

    Returns a list[str] (JSON) per generated step:
      [ "[[token, logprob], [token, logprob], ...]",  ... ]
    """
    out = []
    lp = choice.get("logprobs") or {}
    content = lp.get("content") or lp.get("tokens") or []
    for step in content:
        top = step.get("top_logprobs", None)
        pairs = []

        if isinstance(top, dict):
            # {"token": logprob} mapping
            pairs = [(tok, float(logp)) for tok, logp in top.items() if tok is not None]
        elif isinstance(top, list):
            # [{"token": "str", "logprob": float}, ...]
            for e in top:
                if isinstance(e, dict):
                    tok = e.get("token") or e.get("text") or e.get("token_str")
                    lpv = e.get("logprob") or e.get("log_prob") or e.get("logp")
                    if tok is not None and lpv is not None:
                        try:
                            pairs.append((tok, float(lpv)))
                        except Exception:
                            pass
        else:
            # Unknown shape; skip gracefully
            pairs = []

        # sort by logprob desc if we have numeric values
        try:
            pairs.sort(key=lambda kv: kv[1], reverse=True)
        except Exception:
            pass

        out.append(json.dumps(pairs, ensure_ascii=False))

    return out


# -------------------------
# Arrow read / write utils
# -------------------------
Q_SCHEMA = pa.schema([
    pa.field("qid", pa.int64()),
    pa.field("question_text", pa.string()),
    pa.field("meta", pa.string()),
])

A_SCHEMA = pa.schema([
    pa.field("qid", pa.int64()),
    pa.field("question_text", pa.string()),
    pa.field("answer_text", pa.string()),
    pa.field("topk_json_per_token", pa.list_(pa.string())),
    pa.field("student_input_ids", pa.list_(pa.int32())),
    pa.field("student_role_ids", pa.list_(pa.int8())),
    pa.field("student_loss_mask", pa.list_(pa.int8())),
    pa.field("meta", pa.string()),
])

def _table_to_rows(tbl: pa.Table) -> List[Dict[str, Any]]:
    cols = {n: tbl[n].to_pylist() for n in tbl.column_names}
    rows = []
    n = len(next(iter(cols.values()))) if cols else 0
    for i in range(n):
        rows.append({k: cols[k][i] for k in cols})
    return rows

def read_arrow_if_exists(path: Path, schema: pa.Schema) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with pa_ipc.open_file(path.open("rb")) as reader:
        tbl = reader.read_all()
    # sanity: schema alignment (lenient)
    return _table_to_rows(tbl)

def write_questions_arrow(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pydict({
        "qid": pa.array([r["qid"] for r in rows], type=pa.int64()),
        "question_text": pa.array([r["question_text"] for r in rows], type=pa.string()),
        "meta": pa.array([json.dumps(r.get("meta", {}), ensure_ascii=False) for r in rows], type=pa.string()),
    }, schema=Q_SCHEMA)
    with pa_ipc.new_file(path.open("wb"), Q_SCHEMA) as writer:
        writer.write_table(table)

def write_answers_arrow(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    def arr(name, typ):
        return pa.array([r[name] for r in rows], type=typ)

    table = pa.Table.from_pydict({
        "qid": arr("qid", pa.int64()),
        "question_text": arr("question_text", pa.string()),
        "answer_text": arr("answer_text", pa.string()),
        "topk_json_per_token": arr("topk_json_per_token", pa.list_(pa.string())),
        "student_input_ids": arr("student_input_ids", pa.list_(pa.int32())),
        "student_role_ids": arr("student_role_ids", pa.list_(pa.int8())),
        "student_loss_mask": arr("student_loss_mask", pa.list_(pa.int8())),
        "meta": pa.array([json.dumps(r.get("meta", {}), ensure_ascii=False) for r in rows], type=pa.string()),
    }, schema=A_SCHEMA)
    with pa_ipc.new_file(path.open("wb"), A_SCHEMA) as writer:
        writer.write_table(table)


# ------------------------------------------------------
# Student-side encoding with explicit role/loss masking
# ------------------------------------------------------
def encode_with_roles(tokenizer, user_tok: str, ai_tok: str, question: str, answer: str):
    """
    Build: "[USER] {question}\n[AI] {answer}"
    Tokenize with add_special_tokens=False, so spans are exact.
    Returns (ids, role_ids, loss_mask):
      role_ids: 1=user, 2=assistant, 0=other
      loss_mask: 1=learn (assistant), 0=ignore (user/newline)
    """
    user_part = f"{user_tok} " + question.strip()
    ai_part   = f"{ai_tok} " + answer.strip()
    full_text = user_part + "\n" + ai_part

    enc_full = tokenizer(full_text, add_special_tokens=False)
    ids = enc_full["input_ids"]

    enc_user = tokenizer(user_part, add_special_tokens=False)
    user_len = len(enc_user["input_ids"])

    enc_nl = tokenizer("\n", add_special_tokens=False)
    nl_len = len(enc_nl["input_ids"])

    total_len = len(ids)
    role_ids  = [1]*user_len + [0]*nl_len + [2]*(total_len - user_len - nl_len)
    loss_mask = [0]*user_len + [0]*nl_len + [1]*(total_len - user_len - nl_len)
    return ids, role_ids, loss_mask


# --------------------------------
# Stage A: QUESTION GENERATION
# --------------------------------
def _make_question_request(cfg) -> str:
    """Request a single question; prefer JSON mode, fallback to text."""
    system = {"role": "system", "content": cfg.question_system}
    user_prompt = cfg.question_user.strip() + '\nReturn ONLY {"question": "..."}'
    user = {"role": "user", "content": user_prompt}

    # Try JSON mode
    try:
        resp = chat_complete(
            base_url=cfg.model_base_url,
            api_key=cfg.api_key,
            model=cfg.model_name,
            messages=[system, user],
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=128,
            logprobs=True,
            top_logprobs=int(cfg.top_logprobs),
            prompt_logprobs=int(cfg.prompt_logprobs),
            response_format={"type": "json_object"},
            timeout=120,
        )
        obj = json.loads(resp["choices"][0]["message"]["content"])
        q = obj.get("question", "").strip()
        if q:
            return q
    except Exception:
        pass

    # Fallback: plain text
    resp = chat_complete(
        base_url=cfg.model_base_url,
        api_key=cfg.api_key,
        model=cfg.model_name,
        messages=[system, user],
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_tokens=128,
        logprobs=False,
        top_logprobs=0,
        prompt_logprobs=cfg.prompt_logprobs,
        response_format=None,
        timeout=120,
    )
    text = resp["choices"][0]["message"]["content"]
    for line in text.splitlines():
        s = line.strip().strip("- ").strip()
        if s:
            return s
    return text.strip()

def generate_questions(cfg, qid_start: int) -> List[Dict[str, Any]]:
    total = int(cfg.num_questions_total)
    per_chunk = int(cfg.questions_per_chunk)
    par_calls = int(getattr(cfg, "parallel_calls", getattr(cfg, "parrallel_calls", 4)))

    questions = []
    qid = qid_start
    while len(questions) < total:
        need = min(per_chunk, total - len(questions))
        with ThreadPoolExecutor(max_workers=par_calls) as ex:
            futs = [ex.submit(_make_question_request, cfg) for _ in range(need)]
            for fut in as_completed(futs):
                try:
                    q = fut.result()
                except Exception:
                    q = None
                if q:
                    questions.append({"qid": qid, "question_text": q, "meta": {"src": "teacher_1x"}})
                    qid += 1
    return questions


# --------------------------------
# Stage B: ANSWER GENERATION
# --------------------------------
def _answer_one(teacher_cfg, tokenizer, user_tok, ai_tok, rec) -> Dict[str, Any]:
    system = {"role": "system", "content": teacher_cfg.answer_system}
    user = {"role": "user", "content": rec["question_text"]}

    resp = chat_complete(
        base_url=teacher_cfg.model_base_url,
        api_key=teacher_cfg.api_key,
        model=teacher_cfg.model_name,
        messages=[system, user],
        temperature=teacher_cfg.temperature,
        top_p=teacher_cfg.top_p,
        max_tokens=int(teacher_cfg.max_new_tokens_answer),
        logprobs=True,
        top_logprobs=int(teacher_cfg.top_logprobs),
        prompt_logprobs=int(teacher_cfg.prompt_logprobs),
        response_format=None,
        timeout=180,
    )
    ch = resp["choices"][0]
    ans = ch["message"]["content"]
    topk_json_per_token = parse_topk_json(ch)

    ids, role_ids, loss_mask = encode_with_roles(
        tokenizer, user_tok, ai_tok, rec["question_text"], ans
    )
    return {
        "qid": rec["qid"],
        "question_text": rec["question_text"],
        "answer_text": ans,
        "topk_json_per_token": topk_json_per_token,
        "student_input_ids": [int(x) for x in ids],
        "student_role_ids": [int(min(2, max(0, x))) for x in role_ids],
        "student_loss_mask": [int(min(1, max(0, x))) for x in loss_mask],
        "meta": {
            "model": teacher_cfg.model_name,
            "temperature": float(teacher_cfg.temperature),
            "top_p": float(teacher_cfg.top_p),
        },
    }

def generate_answers(global_cfg, teacher_cfg, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    tok_source, cache_dir = _resolve_student_tokenizer(global_cfg, teacher_cfg)
    tokenizer = AutoTokenizer.from_pretrained(tok_source, use_fast=True, cache_dir=cache_dir)
    user_tok = getattr(teacher_cfg, "user_token", "[USER]")
    ai_tok = getattr(teacher_cfg, "ai_token", "[AI]")

    par_calls = int(getattr(teacher_cfg, "parallel_calls", getattr(teacher_cfg, "parrallel_calls", 4)))
    rows = []
    with ThreadPoolExecutor(max_workers=par_calls) as ex:
        futs = [
            ex.submit(_answer_one, teacher_cfg, tokenizer, user_tok, ai_tok, q)
            for q in questions
        ]
        for fut in as_completed(futs):
            try:
                item = fut.result()
                rows.append(item)
            except Exception as e:
                sys.stderr.write(f"[warn] answer generation failed: {e}\n")
    rows.sort(key=lambda r: r["qid"])
    return rows


# --------------
# Main routine
# --------------
def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default=str(DATA_DIR / "Config.yml"),
        help="Path to teacher_data/Config.yml",
    )
    args = ap.parse_args()

    cfg = _load_config(args.config)
    teacher_cfg = cfg.teacher
    outputs_cfg = cfg.outputs

    base_prefix_str = cfg.paths.get("data_root", "") if "paths" in cfg else ""
    base_prefix = Path(base_prefix_str) if base_prefix_str else None
    if base_prefix is not None and not base_prefix.is_absolute():
        base_prefix = (PROJECT_ROOT / base_prefix).resolve()

    out_dir = _resolve_output_path(base_prefix, outputs_cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    q_path = _resolve_output_path(base_prefix, outputs_cfg.questions_arrow)
    a_path = _resolve_output_path(base_prefix, outputs_cfg.answers_arrow)

    # 0) Read existing (for stacking)
    old_q = read_arrow_if_exists(q_path, Q_SCHEMA)
    old_a = read_arrow_if_exists(a_path, A_SCHEMA)

    # 1) Determine starting qid (continue after current max)
    if old_q:
        max_qid = max(r["qid"] for r in old_q)
        qid_start = int(max_qid) + 1
    else:
        qid_start = 0

    # 2) Generate NEW questions
    new_q = generate_questions(teacher_cfg, qid_start=qid_start)

    # 3) Generate NEW answers for those new questions
    new_a = generate_answers(cfg, teacher_cfg, new_q)

    # 4) Stack and write back (OLD + NEW)
    all_q = (old_q + new_q) if old_q else new_q
    all_a = (old_a + new_a) if old_a else new_a

    write_questions_arrow(q_path, all_q)
    print(f"[ok] wrote {len(new_q)} new questions, total={len(all_q)} → {q_path}")

    write_answers_arrow(a_path, all_a)
    print(f"[ok] wrote {len(new_a)} new QA rows, total={len(all_a)} → {a_path}")

if __name__ == "__main__":
    main()
