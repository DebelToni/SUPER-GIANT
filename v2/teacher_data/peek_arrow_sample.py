#!/usr/bin/env python3
# peek_arrow_sample.py
# Prints a random question + its answer + a peek at top-k logprobs

import argparse, os, random, glob, json
from pathlib import Path
import pyarrow as pa
import pyarrow.ipc as pa_ipc

def find_arrow(path_like: str) -> str:
    p = Path(path_like)
    if p.is_dir():
        # pick first *.arrow in the directory
        cands = sorted(glob.glob(str(p / "*.arrow")))
        if not cands:
            raise FileNotFoundError(f"No .arrow files in directory: {p}")
        return cands[0]
    if p.is_file():
        return str(p)
    raise FileNotFoundError(f"Path not found: {p}")

def read_arrow_table(path: str) -> pa.Table:
    with pa_ipc.open_file(open(path, "rb")) as reader:
        return reader.read_all()

def to_pylist(table: pa.Table):
    return {name: table[name].to_pylist() for name in table.column_names}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="teacher_out/questions.arrow",
                    help="Path OR directory containing questions.arrow")
    ap.add_argument("--answers",   default="teacher_out/answers.arrow",
                    help="Path OR directory containing answers.arrow")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--top_steps", type=int, default=3, help="how many generated steps to show")
    ap.add_argument("--top_k", type=int, default=5, help="how many top-k entries per step to show")
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    q_path = find_arrow(args.questions)
    a_path = find_arrow(args.answers)

    q_tbl = read_arrow_table(q_path)
    a_tbl = read_arrow_table(a_path)
    q = to_pylist(q_tbl)
    a = to_pylist(a_tbl)

    n_q = len(q["qid"]) if "qid" in q else 0
    n_a = len(a["qid"]) if "qid" in a else 0
    if n_q == 0 or n_a == 0:
        raise RuntimeError(f"Empty tables? questions={n_q}, answers={n_a}")

    # choose a random answer row (has both Q & A)
    idx = random.randrange(n_a)

    print("=== BASIC INFO ===")
    print(f"questions file : {q_path}  (rows: {n_q})")
    print(f"answers file   : {a_path}  (rows: {n_a})")
    print(f"sample index   : {idx}")
    print()

    qid = a["qid"][idx]
    q_text = a["question_text"][idx]
    a_text = a["answer_text"][idx]
    print("=== QA SAMPLE ===")
    print(f"QID: {qid}")
    print(f"QUESTION:\n{q_text}\n")
    print(f"ANSWER (first 600 chars):\n{a_text[:600]}\n")

    # peek at student features
    ids = a.get("student_input_ids", [None])[idx]
    role_ids = a.get("student_role_ids", [None])[idx]
    loss_mask = a.get("student_loss_mask", [None])[idx]
    if ids is not None and role_ids is not None and loss_mask is not None:
        print("=== STUDENT FEATURES ===")
        print(f"input_ids len  : {len(ids)}")
        print(f"role_ids  (1=user, 2=assistant)  sample: {role_ids[:16]} ...")
        print(f"loss_mask (1=learn, 0=ignore)    sample: {loss_mask[:16]} ...\n")

    # peek at top-k logprobs (stored as JSON per generated step)
    tk_steps = a.get("topk_json_per_token", [None])[idx]
    if tk_steps:
        steps_to_show = min(args.top_steps, len(tk_steps))
        print("=== TOP-K LOGPROBS (per generated step) ===")
        for t in range(steps_to_show):
            try:
                items = json.loads(tk_steps[t])  # list of [token, logprob] pairs
            except Exception:
                items = []
            print(f"step {t}: ", end="")
            if not items:
                print("(no data)")
                continue
            # show top-K entries
            k = min(args.top_k, len(items))
            preview = ", ".join([f"{tok!r}:{lp:.3f}" for tok, lp in items[:k]])
            print(preview)
        print()

    # sanity: try to cross-check the same qid in questions table
    if "qid" in q and "question_text" in q:
        # lookup by qid (linear scan is fine for debug)
        try:
            pos = q["qid"].index(qid)
            if q["question_text"][pos] != q_text:
                print("WARNING: question text mismatch between files (qid match but text differs).")
        except ValueError:
            print("WARNING: qid from answers not found in questions table.")

    print("Done. Looks consistent if no warnings were printed.")

if __name__ == "__main__":
    main()

