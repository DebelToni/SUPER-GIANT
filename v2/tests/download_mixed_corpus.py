
#!/usr/bin/env python3
import argparse
from pathlib import Path
from datasets import load_dataset

def ensure(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def save(out_dir: Path, prefix: str, idx: int, text: str):
    fname = out_dir / f"{prefix}_{idx:07d}.txt"
    with open(fname, "w", encoding="utf-8", errors="ignore") as f:
        f.write(text)
    return idx + 1

def download_wikipedia(dataset_id: str, prefix: str, max_docs: int, out_dir: Path):
    print(f"Downloading {dataset_id} ...")
    ensure(out_dir)

    ds = load_dataset(dataset_id, split="train", streaming=True)

    idx = 0
    for row in ds:
        text = row.get("text") or row.get("content") or ""
        if text.strip():
            idx = save(out_dir, prefix, idx, text)

        if idx % 500 == 0 and idx > 0:
            print(f"  saved {idx} docs for {prefix}")

        if max_docs is not None and idx >= max_docs:
            break

    print(f"Completed {prefix}: {idx} files.")

def download_python(max_docs: int, out_dir: Path):
    print("Downloading bigcode/the-stack-v2-python ...")
    ensure(out_dir)

    ds = load_dataset("bigcode/the-stack-v2-python", split="train", streaming=True)

    idx = 0
    for row in ds:
        text = row.get("content") or row.get("text") or ""
        if text.strip():
            idx = save(out_dir, "code", idx, text)

        if idx % 500 == 0 and idx > 0:
            print(f"  saved {idx} code files")

        if max_docs is not None and idx >= max_docs:
            break

    print(f"Completed code: {idx} files.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="bg_en_code_final")
    parser.add_argument("--max-bg", type=int, default=5000)
    parser.add_argument("--max-en", type=int, default=5000)
    parser.add_argument("--max-code", type=int, default=5000)
    args = parser.parse_args()

    out = Path(args.output_dir)
    ensure(out)

    max_bg = None if args.max_bg < 0 else args.max_bg
    max_en = None if args.max_en < 0 else args.max_en
    max_code = None if args.max_code < 0 else args.max_code

    download_wikipedia("word-scarcity/wikipedia-bg-20230601", "bg", max_bg, out / "bulgarian")
    download_wikipedia("word-scarcity/wikipedia-en-20230601", "en", max_en, out / "english")
    download_python(max_code, out / "code")

    print("\nAll done.")

if __name__ == "__main__":
    main()
