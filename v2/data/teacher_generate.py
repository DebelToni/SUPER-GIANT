# teacher_generate_omegaconf.py
# Two-stage pipeline:
#   A) Generate single questions via parallel calls (fast & reliable)
#   B) Answer each question, returning top-k logprobs for generated tokens
#
# Writes two Arrow files:
#   - questions.arrow
#   - answers.arrow (with student_input_ids, role_ids, loss_mask)
#
# REQUIREMENTS:
#   pip install omegaconf requests pyarrow "transformers>=4.43.0"
#
# NOTES:
# - Uses vLLM's OpenAI-compatible /chat/completions.
# - We request top-k *logprobs* (not full logits) per generated token.
# - We do NOT rely on tokenizer post-processing; we build "[USER] ... \n[AI] ..." ourselves
#   and run tokenizer with add_special_tokens=False so we can precisely build role/loss masks.

import os, sys, json, time, uuid, math, random
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from omegaconf import OmegaConf
import pyarrow as pa
import pyarrow.ipc as pa_ipc
from transformers import AutoTokenizer

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
    # CHANGED ↓
    top_logprobs: int = 0,
    logprobs: bool = None,
    prompt_logprobs: int = 0,
    response_format: Dict[str, Any] = None,
    seed: int = None,
    timeout: int = 120,
):
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }

    # ---- FIX: OpenAI/vLLM chat schema for logprobs
    if top_logprobs and int(top_logprobs) > 0:
        payload["logprobs"] = True if logprobs is None else bool(logprobs)
        payload["top_logprobs"] = int(top_logprobs)
    elif logprobs is True:
        payload["logprobs"] = True

    # Optional: only include if you truly need prompt token logprobs
    if prompt_logprobs and int(prompt_logprobs) > 0:
        # vLLM supports prompt_logprobs on chat; type handling differs by version.
        # Most recent builds accept an integer here.
        payload["prompt_logprobs"] = int(prompt_logprobs)

    if response_format:
        payload["response_format"] = response_format
    # if seed is not None:
    #     payload["seed"] = int(seed)

    r = requests.post(url, headers=headers, data=_jdump(payload), timeout=timeout)
    r.raise_for_status()
    return r.json()

# --------------------------------------------------
# Parse vLLM chat logprobs into compact JSON strings
# --------------------------------------------------
def parse_topk_json(choice: Dict[str, Any]) -> List[str]:
    """
    Returns a list[str], one JSON-serialized list per generated step: [[(token, logprob), ...], ...]
    vLLM format is choice['logprobs']['content'][t]['top_logprobs'] (dict token->logprob).
    """
    out = []
    lp = choice.get("logprobs", {}) or {}
    content = lp.get("content") or lp.get("tokens") or []
    for step in content:
        top = step.get("top_logprobs") or {}
        # sort by logprob desc for stability
        items = sorted(top.items(), key=lambda kv: kv[1], reverse=True)
        out.append(json.dumps(items, ensure_ascii=False))
    return out

# ----------------------------------------------
# OmegaConf helpers and safe integer conveniences
# ----------------------------------------------
def get_int(cfg, key, default):
    try:
        v = cfg.get(key, default)
    except Exception:
        v = default
    return int(v)

def get_float(cfg, key, default):
    try:
        v = cfg.get(key, default)
    except Exception:
        v = default
    return float(v)

def get_bool(cfg, key, default):
    try:
        v = cfg.get(key, default)
    except Exception:
        v = default
    return bool(v)

# -------------------------
# Arrow writing conveniences
# -------------------------
def write_questions_arrow(path: Path, rows: List[Dict[str, Any]]):
    schema = pa.schema([
        pa.field("qid", pa.int64()),
        pa.field("question_text", pa.string()),
        pa.field("meta", pa.string()),
    ])
    table = pa.Table.from_pydict({
        "qid": pa.array([r["qid"] for r in rows], type=pa.int64()),
        "question_text": pa.array([r["question_text"] for r in rows], type=pa.string()),
        "meta": pa.array([json.dumps(r.get("meta", {}), ensure_ascii=False) for r in rows], type=pa.string()),
    }, schema=schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pa_ipc.new_file(path.open("wb"), schema) as writer:
        writer.write_table(table)

def write_answers_arrow(path: Path, rows: List[Dict[str, Any]]):
    schema = pa.schema([
        pa.field("qid", pa.int64()),
        pa.field("question_text", pa.string()),
        pa.field("answer_text", pa.string()),
        pa.field("topk_json_per_token", pa.list_(pa.string())),
        pa.field("student_input_ids", pa.list_(pa.int32())),
        pa.field("student_role_ids", pa.list_(pa.int8())),
        pa.field("student_loss_mask", pa.list_(pa.int8())),
        pa.field("meta", pa.string()),
    ])
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
    }, schema=schema)

    path.parent.mkdir(parents=True, exist_ok=True)
    with pa_ipc.new_file(path.open("wb"), schema) as writer:
        writer.write_table(table)

# ------------------------------------------------------
# Student-side encoding with explicit role/loss masking
# ------------------------------------------------------
def encode_with_roles(tokenizer, user_tok: str, ai_tok: str, question: str, answer: str):
    """
    Build text as: "[USER] {question}\n[AI] {answer}"
    Then tokenize with add_special_tokens=False, so we fully control indices/spans.
    Returns: (ids[int], role_ids[int8], loss_mask[int8])
      - role_ids: 1 for user, 2 for assistant, 0 elsewhere
      - loss_mask: 1 for assistant tokens, 0 for user/newline
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
    role_user = [1] * user_len
    role_nl   = [0] * nl_len
    role_ai   = [2] * (total_len - user_len - nl_len)
    role_ids  = role_user + role_nl + role_ai

    loss_user = [0] * user_len
    loss_nl   = [0] * nl_len
    loss_ai   = [1] * (total_len - user_len - nl_len)
    loss_mask = loss_user + loss_nl + loss_ai

    return ids, role_ids, loss_mask

# --------------------------------
# Stage A: QUESTION GENERATION
# --------------------------------
def _make_question_request(cfg, seed=None) -> str:
    """
    Request a single question from the teacher.
    Uses JSON mode if supported; falls back to raw text parse otherwise.
    """
    system = {"role": "system", "content": cfg.question_system}
    # Ask explicitly for JSON with a single key "question"
    user_prompt = (
        cfg.question_user.strip()
        + "\nReturn ONLY a JSON object: {\"question\": \"...\"}"
    )
    user = {"role": "user", "content": user_prompt}

    try:
        resp = chat_complete(
            base_url=cfg.model_base_url,
            api_key=cfg.api_key,
            model=cfg.model_name,
            messages=[system, user],
            temperature=float(cfg.temperature),
            top_p=float(cfg.top_p),
            max_tokens=int(cfg.max_new_tokens_answer),
            # NEW (correct schema)
            logprobs=True,
            top_logprobs=int(cfg.top_logprobs),
            prompt_logprobs=int(cfg.prompt_logprobs),
            response_format=None,
            # seed=123,
            timeout=180,
        )
        text = resp["choices"][0]["message"]["content"]
        # parse json
        obj = json.loads(text)
        q = obj.get("question", "").strip()
        if q:
            return q
    except Exception:
        # fall through to non-json mode / relaxed parse
        pass

    # Fallback: try again without response_format and parse text line
    resp = chat_complete(
        base_url=cfg.model_base_url,
        api_key=cfg.api_key,
        model=cfg.model_name,
        messages=[system, user],
        temperature=float(cfg.temperature),
        top_p=float(cfg.top_p),
        max_tokens=128,
        logprobs=0,
        prompt_logprobs=int(cfg.prompt_logprobs),
        response_format=None,
        # seed=seed if seed is not None else 43,
        timeout=120,
    )
    text = resp["choices"][0]["message"]["content"]
    # Heuristic: take first non-empty line as the question
    for line in text.splitlines():
        s = line.strip().strip("- ").strip()
        if s:
            return s
    return text.strip()

def generate_questions(cfg) -> List[Dict[str, Any]]:
    total = int(cfg.num_questions_total)
    per_chunk = int(cfg.questions_per_chunk)
    # accept both 'parrallel_calls' and 'parallel_calls'
    par_calls = cfg.get("parrallel_calls", None)
    if par_calls is None:
        par_calls = cfg.get("parallel_calls", 4)
    par_calls = int(par_calls)

    questions = []
    qid = 0
    while len(questions) < total:
        remaining = total - len(questions)
        chunk_n = min(per_chunk, remaining)
        seeds = [random.randint(1, 10_000_000) for _ in range(chunk_n)]

        with ThreadPoolExecutor(max_workers=par_calls) as ex:
            futs = [ex.submit(_make_question_request, cfg, seeds[i]) for i in range(chunk_n)]
            for fut in as_completed(futs):
                try:
                    q = fut.result()
                except Exception as e:
                    q = None
                if q:
                    questions.append({"qid": qid, "question_text": q, "meta": {"src": "teacher_1x"}})
                    qid += 1

    return questions[:total]

# --------------------------------
# Stage B: ANSWER GENERATION
# --------------------------------
def _answer_one(cfg, tokenizer, user_tok, ai_tok, rec) -> Dict[str, Any]:
    # Build chat messages for answering
    system = {"role": "system", "content": cfg.answer_system}
    user = {"role": "user", "content": rec["question_text"]}

    resp = chat_complete(
        base_url=cfg.model_base_url,
        api_key=cfg.api_key,
        model=cfg.model_name,
        messages=[system, user],
        temperature=float(cfg.temperature),
        top_p=float(cfg.top_p),
        max_tokens=int(cfg.max_new_tokens_answer),
        logprobs=int(cfg.top_logprobs),
        prompt_logprobs=int(cfg.prompt_logprobs),
        response_format=None,
        # seed=123,
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
        "meta": {"model": cfg.model_name},
    }

def generate_answers(cfg, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(cfg.student_tokenizer_dir, use_fast=True)
    # We assume you've already added [USER] and [AI] to this tokenizer.
    user_tok = cfg.user_token
    ai_tok = cfg.ai_token

    par_calls = cfg.get("parrallel_calls", None)
    if par_calls is None:
        par_calls = cfg.get("parallel_calls", 4)
    par_calls = int(par_calls)

    rows = []
    with ThreadPoolExecutor(max_workers=par_calls) as ex:
        futs = [ex.submit(_answer_one, cfg, tokenizer, user_tok, ai_tok, q) for q in questions]
        for fut in as_completed(futs):
            try:
                item = fut.result()
                rows.append(item)
            except Exception as e:
                # log and continue
                sys.stderr.write(f"[warn] answer generation failed: {e}\n")
    # keep sorted by qid
    rows.sort(key=lambda r: r["qid"])
    return rows

# --------------
# Main routine
# --------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to Teacher_config.yml")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stage A: Questions
    questions = generate_questions(cfg)
    write_questions_arrow(Path(cfg.questions_arrow), questions)
    print(f"[ok] wrote {len(questions)} questions → {cfg.questions_arrow}")

    # Stage B: Answers (+ role & loss masks)
    answers = generate_answers(cfg, questions)
    write_answers_arrow(Path(cfg.answers_arrow), answers)
    print(f"[ok] wrote {len(answers)} QA rows → {cfg.answers_arrow}")

if __name__ == "__main__":
    main()

