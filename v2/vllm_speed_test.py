import asyncio
import os
import time
import httpx

# ---- CONFIG ----
# BASE_URL = "http://127.0.0.1:8000/v1/completions"
DEFAULT_BASE_URL = "https://rh3d0k2emx2ntt-8000.proxy.runpod.net/v1/completions"
BASE_ROOT = os.getenv("BASE")
BASE_URL = os.getenv("VLLM_BASE_URL") or (
    f"{BASE_ROOT}/v1/completions" if BASE_ROOT else DEFAULT_BASE_URL
)
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
CONCURRENCY =  8
N_TOKENS = 512          # <-- set your N here
REQUESTS_PER_USER = 4   # bump this to average over more runs
PROMPT = "Write a technical explenation about GPU memory bandwidth.\n"
API_KEY = os.getenv("VLLM_API_KEY")
# ---------------

async def one_request(client: httpx.AsyncClient) -> int:
    payload = {
        "model": MODEL,
        "prompt": PROMPT,
        "max_tokens": N_TOKENS,
        "temperature": 0.0,
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else None
    r = await client.post(BASE_URL, json=payload, headers=headers, timeout=None)
    r.raise_for_status()
    j = r.json()
    # OpenAI-style usage field
    usage = j.get("usage") or {}
    completion_tokens = int(usage.get("completion_tokens") or 0)

    # Fallback if usage missing for some reason
    if completion_tokens == 0:
        text = (j.get("choices") or [{}])[0].get("text", "")
        # crude fallback: count whitespace-separated tokens
        completion_tokens = len(text.split())

    return completion_tokens

async def user_loop(client: httpx.AsyncClient) -> int:
    total = 0
    for _ in range(REQUESTS_PER_USER):
        total += await one_request(client)
    return total

async def main():
    async with httpx.AsyncClient() as client:
        t0 = time.perf_counter()
        totals = await asyncio.gather(*[user_loop(client) for _ in range(CONCURRENCY)])
        t1 = time.perf_counter()

    total_tokens = sum(totals)
    elapsed = t1 - t0
    tps = total_tokens / elapsed if elapsed > 0 else float("inf")

    print(f"concurrency={CONCURRENCY}")
    print(f"tokens_per_request={N_TOKENS}")
    print(f"requests_per_user={REQUESTS_PER_USER}")
    print(f"total_completion_tokens={total_tokens}")
    print(f"elapsed_s={elapsed:.4f}")
    print(f"throughput_tokens_per_s={tps:.2f}")

if __name__ == "__main__":
    asyncio.run(main())
