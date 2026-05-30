from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from typing import List

import jax
import jax.numpy as jnp
import numpy as np

from GIANT.v3.model.Generate_faster import (
    align_tokenizer_and_params_vocab,
    build_model,
    choose_bucketed_context_length,
    default_auto_buckets,
    load_configs,
    load_params,
    load_tokenizer,
    normalize_buckets,
    parse_bool_flag,
    parse_int_list,
    resolve_checkpoint_path,
)
from GIANT.v3.model.jit_inference import init_inference_state, make_prefill_and_decode_fns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat-style generation for SUPER-GIANT checkpoints.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--global_config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="latest")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--system", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None, help="Optional first user turn.")
    parser.add_argument("--user", type=str, action="append", default=[])
    parser.add_argument(
        "--message",
        type=str,
        action="append",
        default=[],
        help="Explicit message in the form role:text. Can be passed multiple times.",
    )
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_context", "--context_length", type=int, default=None, dest="max_context")
    parser.add_argument("--stop_on_eos", type=str, default=None)
    parser.add_argument("--kv_cache_buckets", type=str, default=None)
    parser.add_argument("--disable_kv_buckets", action="store_true")
    parser.add_argument("--interactive", action="store_true", help="Continue in stdin/stdout chat mode after any initial turns.")
    parser.add_argument("--window", action="store_true", help="Open a simple desktop chat window instead of stdin/stdout interaction.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def clone_state(tree):
    return jax.tree_util.tree_map(lambda x: jnp.array(x, copy=True), tree)


def generate_tokens(
    *,
    params,
    base_state,
    prefill_fn,
    decode_fn,
    prompt_ids: np.ndarray,
    steps: int,
    temperature: float,
    top_k: int,
    do_sample: bool,
    rng_key: jax.random.KeyArray,
):
    state = clone_state(base_state)
    prompt = jnp.asarray(prompt_ids[None, :], dtype=jnp.int32)

    prefill_start = time.perf_counter()
    nonparam, t_cur, last_tok = prefill_fn(params, state, prompt)
    for leaf in jax.tree_util.tree_leaves((nonparam, last_tok)):
        if isinstance(leaf, jax.Array):
            leaf.block_until_ready()
    prefill_time = time.perf_counter() - prefill_start

    rng = None
    new_rng = rng_key
    if do_sample:
        new_rng, rng = jax.random.split(rng_key)

    decode_start = time.perf_counter()
    tokens_new, _ = decode_fn(
        params,
        nonparam,
        last_tok,
        t_cur,
        steps=steps,
        do_sample=do_sample,
        top_k=top_k,
        temperature=temperature,
        rng_key=rng,
    )
    tokens_new.block_until_ready()
    decode_time = time.perf_counter() - decode_start
    return np.asarray(tokens_new[0]), prefill_time, decode_time, new_rng


def _parse_messages(args: argparse.Namespace) -> List[dict[str, str]]:
    messages: List[dict[str, str]] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    if args.prompt:
        messages.append({"role": "user", "content": args.prompt})
    for user_text in args.user:
        messages.append({"role": "user", "content": user_text})
    for raw in args.message:
        role, sep, content = raw.partition(":")
        if not sep:
            raise ValueError(f"Invalid --message value {raw!r}; expected role:text")
        role = role.strip()
        content = content.strip()
        if not role or not content:
            raise ValueError(f"Invalid --message value {raw!r}; expected role:text")
        messages.append({"role": role, "content": content})
    return messages


def _base_messages_for_new_context(args: argparse.Namespace) -> List[dict[str, str]]:
    base_messages: List[dict[str, str]] = []
    if args.system:
        base_messages.append({"role": "system", "content": args.system})
    return base_messages


def _render_chat_prompt(tokenizer, messages: List[dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except (ImportError, ValueError):
            pass
    parts: List[str] = []
    for message in messages:
        parts.append(f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def _truncate_prompt_ids(prompt_ids: List[int], context_length: int) -> np.ndarray:
    if not prompt_ids:
        raise ValueError("Chat prompt produced zero tokens.")
    if len(prompt_ids) >= context_length:
        prompt_ids = prompt_ids[-context_length:]
    return np.asarray(prompt_ids, dtype=np.int32)


def _trim_chat_response(tokenizer, generated_tokens: np.ndarray, stop_on_eos: bool) -> np.ndarray:
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is not None and int(im_end_id) >= 0:
        stop_positions = np.where(generated_tokens == int(im_end_id))[0]
        if stop_positions.size > 0:
            return generated_tokens[: int(stop_positions[0])]
    if stop_on_eos and tokenizer.eos_token_id is not None:
        stop_positions = np.where(generated_tokens == int(tokenizer.eos_token_id))[0]
        if stop_positions.size > 0:
            return generated_tokens[: int(stop_positions[0])]
    return generated_tokens


def _render_text_for_stdout(text: str) -> str:
    rendered = (
        text.replace("\\r\\n", "\n")
        .replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace("<0x0A>", "\n")
        .replace("<0x09>", "\t")
    )
    fence_pattern = re.compile(r"```([A-Za-z0-9_+.-]*)\s+(.+?)\s+```", re.DOTALL)

    def _format_code_body(lang: str, body: str) -> str:
        formatted = body.strip().replace("; ", ";\n")
        if "\n" not in formatted and lang.lower() in {"python", "py"}:
            formatted = re.sub(r":\s+(?=\S)", ":\n    ", formatted)
        return formatted

    def _fence_repl(match: re.Match[str]) -> str:
        lang = match.group(1)
        body = _format_code_body(lang, match.group(2))
        header = f"```{lang}" if lang else "```"
        return f"{header}\n{body}\n```"

    rendered = fence_pattern.sub(_fence_repl, rendered)
    rendered = re.sub(r"(?<!\n)(\d+\.\s+\*\*)", r"\n\1", rendered)
    rendered = re.sub(r"(?<!\n)(-\s+)", r"\n\1", rendered)
    return rendered


def _run_turn(
    *,
    tokenizer,
    messages: List[dict[str, str]],
    context_length: int,
    steps: int,
    stop_on_eos: bool,
    params,
    base_state,
    prefill_fn,
    decode_fn,
    temperature: float,
    top_k: int,
    do_sample: bool,
    sample_key,
) -> tuple[str, float, float, object]:
    prompt_text = _render_chat_prompt(tokenizer, messages)
    prompt_ids = _truncate_prompt_ids(tokenizer.encode(prompt_text, add_special_tokens=False), context_length)
    generated_tokens, prefill_time, decode_time, sample_key = generate_tokens(
        params=params,
        base_state=base_state,
        prefill_fn=prefill_fn,
        decode_fn=decode_fn,
        prompt_ids=prompt_ids,
        steps=steps,
        temperature=temperature,
        top_k=top_k,
        do_sample=do_sample,
        rng_key=sample_key,
    )
    trimmed = _trim_chat_response(tokenizer, generated_tokens, stop_on_eos)
    response_text = _render_text_for_stdout(tokenizer.decode(trimmed, skip_special_tokens=True).strip())
    return response_text, prefill_time, decode_time, sample_key


def _run_browser_window_chat(
    *,
    args: argparse.Namespace,
    checkpoint_path,
    messages: List[dict[str, str]],
    base_messages: List[dict[str, str]],
    has_non_system_seed: bool,
    tokenizer,
    context_length: int,
    max_steps: int,
    stop_on_eos: bool,
    params,
    base_state,
    prefill_fn,
    decode_fn,
    temperature: float,
    top_k: int,
    do_sample: bool,
    sample_key,
) -> None:
    import socketserver
    import threading
    import webbrowser
    from http.server import BaseHTTPRequestHandler
    from urllib.parse import parse_qs, urlparse

    giant_messages = list(messages)
    giant_base_messages = list(base_messages)
    giant_transcript = [
        {"role": ("GIANT" if message["role"] == "assistant" else message["role"].capitalize()), "text": message["content"]}
        for message in giant_messages
    ]
    gpt2_transcript: List[dict[str, str]] = []
    gpt2_worker = None
    gpt2_ready_status = "Mode: GPT-2 | Greedy continuation of 10 tokens"
    lock = threading.Lock()

    def default_status(mode: str) -> str:
        if mode == "GIANT":
            return f"Mode: GIANT | Checkpoint: {checkpoint_path.name}"
        return "Mode: GPT-2 | Greedy continuation of 10 tokens"

    def ensure_gpt2_worker():
        nonlocal gpt2_worker, gpt2_ready_status
        if gpt2_worker is not None and gpt2_worker.poll() is None:
            return gpt2_worker, gpt2_ready_status

        worker_code = r'''
import json
import os
import sys

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.to(device)
model.eval()

print(json.dumps({"ready": True, "device": device}), flush=True)

for raw in sys.stdin:
    if not raw.strip():
        continue
    try:
        payload = json.loads(raw)
        prompt = str(payload.get("prompt", ""))
        encoded = tokenizer(prompt, return_tensors="pt")
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            output = model.generate(
                **encoded,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        continuation = output[0][encoded["input_ids"].shape[1]:]
        text = tokenizer.decode(continuation, skip_special_tokens=True)
        print(json.dumps({"ok": True, "text": text, "device": device}), flush=True)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), flush=True)
'''
        gpt2_worker = subprocess.Popen(
            ["/Volumes/SSD/v/py/bin/python", "-u", "-c", worker_code],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        if gpt2_worker.stdout is None:
            raise RuntimeError("Failed to start GPT-2 worker stdout pipe")
        ready = None
        while True:
            ready_line = gpt2_worker.stdout.readline()
            if ready_line == "":
                break
            ready_line = ready_line.strip()
            if not ready_line:
                continue
            try:
                candidate = json.loads(ready_line)
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict) and candidate.get("ready"):
                ready = candidate
                break
        if ready is None:
            err = gpt2_worker.stderr.read() if gpt2_worker.stderr is not None else ""
            raise RuntimeError(f"GPT-2 worker failed to start: {err.strip()}")
        if not ready.get("ready"):
            raise RuntimeError(f"GPT-2 worker startup failed: {ready}")
        gpt2_ready_status = f"Mode: GPT-2 | Device: {ready.get('device', 'unknown')} | Greedy continuation of 10 tokens"
        return gpt2_worker, gpt2_ready_status

    def reset_mode(mode: str) -> None:
        nonlocal giant_messages, giant_transcript, gpt2_transcript
        if mode == "GIANT":
            giant_messages = list(giant_base_messages)
            giant_transcript = [
                {"role": message["role"].capitalize(), "text": message["content"]}
                for message in giant_messages
            ]
        else:
            gpt2_transcript = []

    def state_for_mode(mode: str, status: str | None = None) -> dict[str, object]:
        transcript = giant_transcript if mode == "GIANT" else gpt2_transcript
        return {"mode": mode, "messages": transcript, "status": status or default_status(mode)}

    def run_gpt2_generation(prompt_text: str) -> str:
        worker, ready_status = ensure_gpt2_worker()
        if worker.stdin is None or worker.stdout is None:
            raise RuntimeError("GPT-2 worker pipes are unavailable")
        worker.stdin.write(json.dumps({"prompt": prompt_text}) + "\n")
        worker.stdin.flush()
        response_line = worker.stdout.readline().strip()
        if not response_line:
            err = worker.stderr.read() if worker.stderr is not None else ""
            raise RuntimeError(f"GPT-2 worker exited unexpectedly: {err.strip()}")
        payload = json.loads(response_line)
        if not payload.get("ok"):
            raise RuntimeError(f"GPT-2 generation failed: {payload.get('error', 'unknown error')}")
        return ready_status + "| ready"

    if giant_messages and has_non_system_seed:
        response_text, _, _, sample_key = _run_turn(
            tokenizer=tokenizer,
            messages=giant_messages,
            context_length=context_length,
            steps=max_steps,
            stop_on_eos=stop_on_eos,
            params=params,
            base_state=base_state,
            prefill_fn=prefill_fn,
            decode_fn=decode_fn,
            temperature=temperature,
            top_k=top_k,
            do_sample=do_sample,
            sample_key=sample_key,
        )
        giant_messages.append({"role": "assistant", "content": response_text})
        giant_transcript.append({"role": "GIANT", "text": response_text})

    page = """<!doctype html>
<html>
<head>
  <meta charset='utf-8'>
  <title>GIANT Demo</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; background: #f8fafc; color: #0f172a; }
    .wrap { max-width: 1180px; margin: 0 auto; padding: 28px; }
    .top { display: flex; gap: 14px; align-items: center; margin-bottom: 14px; }
    select, button, textarea { font: inherit; }
    select, button { padding: 10px 14px; border-radius: 10px; border: 1px solid #cbd5e1; background: #ffffff; color: #0f172a; font-size: 18px; }
    label { font-size: 18px; font-weight: 600; }
    #status { margin: 12px 0 16px; color: #475569; font-size: 18px; }
    #transcript { min-height: 520px; background: #ffffff; border: 1px solid #cbd5e1; border-radius: 16px; padding: 22px; overflow-y: auto; box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06); }
    .msg { margin-bottom: 18px; white-space: pre-wrap; font-size: 20px; line-height: 1.45; }
    .role { font-weight: 700; color: #1d4ed8; margin-bottom: 6px; font-size: 16px; letter-spacing: 0.02em; text-transform: uppercase; }
    textarea { width: 100%; min-height: 150px; margin-top: 16px; border-radius: 16px; border: 1px solid #cbd5e1; background: #ffffff; color: #0f172a; padding: 16px; box-sizing: border-box; font-size: 20px; line-height: 1.45; }
    .hint { color: #64748b; margin-top: 12px; font-size: 16px; }
  </style>
</head>
<body>
  <div class='wrap'>
    <div class='top'>
      <label for='mode'>Model</label>
      <select id='mode'>
        <option value='GIANT'>GIANT</option>
        <option value='GPT-2'>GPT-2</option>
      </select>
      <button onclick='resetChat()'>New Chat</button>
      <button onclick='sendMessage()'>Send</button>
    </div>
    <div id='status'></div>
    <div id='transcript'></div>
    <textarea id='input' placeholder='Type here. Cmd+Enter or Ctrl+Enter to send.'></textarea>
    <div class='hint'>GIANT keeps chat history. GPT-2 is plain continuation mode and greedily generates 10 tokens. Type /new to reset the current mode.</div>
  </div>
  <script>
    const modeEl = document.getElementById('mode');
    const statusEl = document.getElementById('status');
    const transcriptEl = document.getElementById('transcript');
    const inputEl = document.getElementById('input');

    function appendLocalMessage(role, text) {
      const box = document.createElement('div');
      box.className = 'msg';
      const roleEl = document.createElement('div');
      roleEl.className = 'role';
      roleEl.textContent = role;
      const textEl = document.createElement('div');
      textEl.textContent = text;
      box.appendChild(roleEl);
      box.appendChild(textEl);
      transcriptEl.appendChild(box);
      transcriptEl.scrollTop = transcriptEl.scrollHeight;
    }

    function renderState(state) {
      statusEl.textContent = state.status;
      transcriptEl.innerHTML = '';
      for (const msg of state.messages) {
        const box = document.createElement('div');
        box.className = 'msg';
        const role = document.createElement('div');
        role.className = 'role';
        role.textContent = msg.role;
        const text = document.createElement('div');
        text.textContent = msg.text;
        box.appendChild(role);
        box.appendChild(text);
        transcriptEl.appendChild(box);
      }
      transcriptEl.scrollTop = transcriptEl.scrollHeight;
    }

    async function loadState() {
      const res = await fetch('/state?mode=' + encodeURIComponent(modeEl.value));
      renderState(await res.json());
    }

    async function sendMessage() {
      const text = inputEl.value.trim();
      if (!text) return;
      const mode = modeEl.value;
      appendLocalMessage(mode === 'GIANT' ? 'User' : 'Prompt', text);
      statusEl.textContent = mode === 'GIANT' ? 'Generating GIANT response...' : 'Loading GPT-2 and generating continuation...';
      inputEl.value = '';
      const res = await fetch('/send', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode, text })
      });
      const payload = await res.json();
      if (!res.ok) {
        statusEl.textContent = payload.error || 'Request failed';
        await loadState();
        return;
      }
      renderState(payload);
      inputEl.focus();
    }

    async function resetChat() {
      const res = await fetch('/reset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: modeEl.value })
      });
      renderState(await res.json());
      inputEl.value = '';
      inputEl.focus();
    }

    modeEl.addEventListener('change', loadState);
    inputEl.addEventListener('keydown', (event) => {
      if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') {
        event.preventDefault();
        sendMessage();
      }
    });

    loadState();
  </script>
</body>
</html>
"""

    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, payload: dict[str, object], status_code: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/state":
                mode = parse_qs(parsed.query).get("mode", ["GIANT"])[0]
                with lock:
                    self._send_json(state_for_mode(mode))
                return
            if parsed.path == "/":
                body = page.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_error(404)

        def do_POST(self) -> None:  # noqa: N802
            nonlocal sample_key, giant_messages, giant_transcript, gpt2_transcript
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                mode = str(payload.get("mode", "GIANT"))
                text = str(payload.get("text", "")).strip()
                with lock:
                    if self.path == "/reset" or text == "/new":
                        reset_mode(mode)
                        self._send_json(state_for_mode(mode))
                        return
                    if self.path != "/send":
                        self.send_error(404)
                        return
                    if mode == "GIANT":
                        giant_messages.append({"role": "user", "content": text})
                        giant_transcript.append({"role": "User", "text": text})
                        response_text, _, _, sample_key = _run_turn(
                            tokenizer=tokenizer,
                            messages=giant_messages,
                            context_length=context_length,
                            steps=max_steps,
                            stop_on_eos=stop_on_eos,
                            params=params,
                            base_state=base_state,
                            prefill_fn=prefill_fn,
                            decode_fn=decode_fn,
                            temperature=temperature,
                            top_k=top_k,
                            do_sample=do_sample,
                            sample_key=sample_key,
                        )
                        giant_messages.append({"role": "assistant", "content": response_text})
                        giant_transcript.append({"role": "GIANT", "text": response_text})
                        self._send_json(state_for_mode(mode))
                        return

                    worker, ready_status = ensure_gpt2_worker()
                    if worker.stdin is None or worker.stdout is None:
                        raise RuntimeError("GPT-2 worker pipes are unavailable")
                    gpt2_transcript.append({"role": "Prompt", "text": text})
                    worker.stdin.write(json.dumps({"prompt": text}) + "\n")
                    worker.stdin.flush()
                    response_payload = None
                    while True:
                        response_line = worker.stdout.readline()
                        if response_line == "":
                            break
                        response_line = response_line.strip()
                        if not response_line:
                            continue
                        try:
                            candidate = json.loads(response_line)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(candidate, dict) and "ok" in candidate:
                            response_payload = candidate
                            break
                    if response_payload is None:
                        err = worker.stderr.read() if worker.stderr is not None else ""
                        raise RuntimeError(f"GPT-2 worker exited unexpectedly: {err.strip()}")
                    if not response_payload.get("ok"):
                        raise RuntimeError(f"GPT-2 generation failed: {response_payload.get('error', 'unknown error')}")
                    gpt2_transcript.append({"role": "GPT-2", "text": str(response_payload.get('text', '')).strip() or '<no new text>'})
                    self._send_json(state_for_mode(mode, status=ready_status))
            except Exception as exc:  # pragma: no cover - UI error path
                self._send_json({"error": str(exc)}, status_code=500)

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

    class DemoServer(socketserver.ThreadingTCPServer):
        allow_reuse_address = True

    with DemoServer(("127.0.0.1", 0), Handler) as server:
        port = server.server_address[1]
        url = f"http://127.0.0.1:{port}/"
        print(f"Opening demo window at {url}")
        webbrowser.open(url)
        try:
            server.serve_forever()
        finally:
            if gpt2_worker is not None and gpt2_worker.poll() is None:
                gpt2_worker.terminate()
                try:
                    gpt2_worker.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    gpt2_worker.kill()


def _run_window_chat(
    *,
    args: argparse.Namespace,
    checkpoint_path,
    messages: List[dict[str, str]],
    base_messages: List[dict[str, str]],
    has_non_system_seed: bool,
    tokenizer,
    context_length: int,
    max_steps: int,
    stop_on_eos: bool,
    params,
    base_state,
    prefill_fn,
    decode_fn,
    temperature: float,
    top_k: int,
    do_sample: bool,
    sample_key,
) -> None:
    try:
        import tkinter as tk
        from tkinter import scrolledtext, ttk
    except ImportError as exc:  # pragma: no cover - platform dependent
        _run_browser_window_chat(
            args=args,
            checkpoint_path=checkpoint_path,
            messages=messages,
            base_messages=base_messages,
            has_non_system_seed=has_non_system_seed,
            tokenizer=tokenizer,
            context_length=context_length,
            max_steps=max_steps,
            stop_on_eos=stop_on_eos,
            params=params,
            base_state=base_state,
            prefill_fn=prefill_fn,
            decode_fn=decode_fn,
            temperature=temperature,
            top_k=top_k,
            do_sample=do_sample,
            sample_key=sample_key,
        )
        return

    giant_messages = list(messages)
    giant_base_messages = list(base_messages)
    gpt2_worker = None

    root = tk.Tk()
    root.title("GIANT Chat Demo")
    root.geometry("980x720")

    header = tk.Frame(root)
    header.pack(fill=tk.X, padx=12, pady=(12, 8))

    tk.Label(header, text="Model", anchor="w").pack(side=tk.LEFT)
    mode_var = tk.StringVar(value="GIANT")
    mode_box = ttk.Combobox(header, textvariable=mode_var, values=["GIANT", "GPT-2"], state="readonly", width=16)
    mode_box.pack(side=tk.LEFT, padx=(8, 0))

    transcript = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=("Menlo", 13))
    transcript.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 8))
    transcript.configure(state=tk.DISABLED)

    status_var = tk.StringVar(value=f"Mode: GIANT | Checkpoint: {checkpoint_path.name}")
    status = tk.Label(root, textvariable=status_var, anchor="w")
    status.pack(fill=tk.X, padx=12)

    controls = tk.Frame(root)
    controls.pack(fill=tk.X, padx=12, pady=8)

    input_box = tk.Text(controls, height=4, wrap=tk.WORD, font=("Menlo", 13))
    input_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def append_message(role: str, text: str) -> None:
        transcript.configure(state=tk.NORMAL)
        transcript.insert(tk.END, f"{role}: {text}\n\n")
        transcript.configure(state=tk.DISABLED)
        transcript.see(tk.END)

    def clear_transcript() -> None:
        transcript.configure(state=tk.NORMAL)
        transcript.delete("1.0", tk.END)
        transcript.configure(state=tk.DISABLED)

    def current_mode() -> str:
        return mode_var.get()

    def set_default_status() -> None:
        if current_mode() == "GIANT":
            status_var.set(f"Mode: GIANT | Checkpoint: {checkpoint_path.name}")
        else:
            status_var.set("Mode: GPT-2 | Greedy continuation of 10 tokens")

    def set_busy(is_busy: bool, detail: str = "") -> None:
        if is_busy:
            root.config(cursor="watch")
            input_box.config(state=tk.DISABLED)
            mode_box.config(state="disabled")
            send_btn.config(state=tk.DISABLED)
            new_btn.config(state=tk.DISABLED)
            clear_btn.config(state=tk.DISABLED)
            status_var.set(detail or "Generating...")
        else:
            root.config(cursor="")
            input_box.config(state=tk.NORMAL)
            mode_box.config(state="readonly")
            send_btn.config(state=tk.NORMAL)
            new_btn.config(state=tk.NORMAL)
            clear_btn.config(state=tk.NORMAL)
            status_var.set(detail or "")
            if not detail:
                set_default_status()
            input_box.focus_set()
        root.update_idletasks()

    def ensure_gpt2_worker():
        nonlocal gpt2_worker
        if gpt2_worker is not None and gpt2_worker.poll() is None:
            return gpt2_worker

        worker_code = r'''
import json
import os
import sys

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.to(device)
model.eval()

print(json.dumps({"ready": True, "device": device}), flush=True)

for raw in sys.stdin:
    if not raw.strip():
        continue
    try:
        payload = json.loads(raw)
        prompt = str(payload.get("prompt", ""))
        encoded = tokenizer(prompt, return_tensors="pt")
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            output = model.generate(
                **encoded,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        continuation = output[0][encoded["input_ids"].shape[1]:]
        text = tokenizer.decode(continuation, skip_special_tokens=True)
        print(json.dumps({"ok": True, "text": text, "device": device}), flush=True)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), flush=True)
'''
        gpt2_worker = subprocess.Popen(
            ["/Volumes/SSD/v/py/bin/python", "-u", "-c", worker_code],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        if gpt2_worker.stdout is None:
            raise RuntimeError("Failed to start GPT-2 worker stdout pipe")
        ready_line = gpt2_worker.stdout.readline().strip()
        if not ready_line:
            err = gpt2_worker.stderr.read() if gpt2_worker.stderr is not None else ""
            raise RuntimeError(f"GPT-2 worker failed to start: {err.strip()}")
        ready = json.loads(ready_line)
        if not ready.get("ready"):
            raise RuntimeError(f"GPT-2 worker startup failed: {ready}")
        status_var.set(f"Mode: GPT-2 | Device: {ready.get('device', 'unknown')} | Greedy continuation of 10 tokens")
        return gpt2_worker

    def run_giant_generation() -> None:
        nonlocal sample_key, giant_messages
        response_text, prefill_time, decode_time, sample_key = _run_turn(
            tokenizer=tokenizer,
            messages=giant_messages,
            context_length=context_length,
            steps=max_steps,
            stop_on_eos=stop_on_eos,
            params=params,
            base_state=base_state,
            prefill_fn=prefill_fn,
            decode_fn=decode_fn,
            temperature=temperature,
            top_k=top_k,
            do_sample=do_sample,
            sample_key=sample_key,
        )
        giant_messages.append({"role": "assistant", "content": response_text})
        append_message("GIANT", response_text)
        if args.verbose:
            toks_per_s = (max_steps / decode_time) if decode_time > 0 else float("inf")
            append_message(
                "perf",
                f"prefill={prefill_time:.4f}s decode={decode_time:.4f}s tokens/s={toks_per_s:.2f}",
            )

    def run_gpt2_generation(prompt_text: str) -> None:
        worker = ensure_gpt2_worker()
        if worker.stdin is None or worker.stdout is None:
            raise RuntimeError("GPT-2 worker pipes are unavailable")
        worker.stdin.write(json.dumps({"prompt": prompt_text}) + "\n")
        worker.stdin.flush()
        response_line = worker.stdout.readline().strip()
        if not response_line:
            err = worker.stderr.read() if worker.stderr is not None else ""
            raise RuntimeError(f"GPT-2 worker exited unexpectedly: {err.strip()}")
        payload = json.loads(response_line)
        if not payload.get("ok"):
            raise RuntimeError(f"GPT-2 generation failed: {payload.get('error', 'unknown error')}")
        continuation_text = str(payload.get("text", "")).strip()
        append_message("GPT-2", continuation_text or "<no new text>")

    def reset_chat() -> None:
        nonlocal giant_messages
        giant_messages = list(giant_base_messages)
        clear_transcript()
        if current_mode() == "GIANT":
            for message in giant_messages:
                append_message(message["role"].capitalize(), message["content"])
        set_default_status()

    def switch_mode(event=None) -> None:
        reset_chat()

    def close_window() -> None:
        nonlocal gpt2_worker
        if gpt2_worker is not None and gpt2_worker.poll() is None:
            gpt2_worker.terminate()
            try:
                gpt2_worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                gpt2_worker.kill()
        root.destroy()

    def send_message(event=None):
        nonlocal giant_messages
        user_text = input_box.get("1.0", tk.END).strip()
        if not user_text:
            return "break"
        if user_text == "/new":
            reset_chat()
            input_box.delete("1.0", tk.END)
            return "break"
        if user_text == "/clear":
            clear_transcript()
            input_box.delete("1.0", tk.END)
            return "break"

        input_box.delete("1.0", tk.END)
        mode = current_mode()
        append_message("User" if mode == "GIANT" else "Prompt", user_text)
        if mode == "GIANT":
            giant_messages.append({"role": "user", "content": user_text})
            busy_text = "Generating GIANT response..."
        else:
            busy_text = "Loading GPT-2 and generating continuation..."
        set_busy(True, busy_text)
        try:
            if mode == "GIANT":
                run_giant_generation()
            else:
                run_gpt2_generation(user_text)
        finally:
            set_busy(False)
        return "break"

    send_btn = tk.Button(controls, text="Send", width=10, command=send_message)
    send_btn.pack(side=tk.LEFT, padx=(8, 0))
    new_btn = tk.Button(controls, text="New Chat", width=10, command=reset_chat)
    new_btn.pack(side=tk.LEFT, padx=(8, 0))
    clear_btn = tk.Button(controls, text="Clear", width=10, command=clear_transcript)
    clear_btn.pack(side=tk.LEFT, padx=(8, 0))

    mode_box.bind("<<ComboboxSelected>>", switch_mode)
    input_box.bind("<Command-Return>", send_message)
    input_box.bind("<Control-Return>", send_message)
    root.protocol("WM_DELETE_WINDOW", close_window)

    for message in giant_messages:
        append_message(message["role"].capitalize(), message["content"])

    if giant_messages and has_non_system_seed:
        set_busy(True, "Generating initial response...")
        try:
            run_giant_generation()
        finally:
            set_busy(False)

    input_box.focus_set()
    root.mainloop()


def main() -> None:
    args = parse_args()
    cfg = load_configs(args.config, args.global_config)
    jax.config.update("jax_default_matmul_precision", cfg.model.compute_dtype)

    cfg_temperature = float(getattr(cfg.inference, "temperature", 0.0))
    cfg_top_k = int(getattr(cfg.inference, "top_k", 0))
    cfg_steps = int(getattr(cfg.inference, "max_decode_steps", 128))
    cfg_stop_on_eos = bool(getattr(cfg.inference, "stop_on_eos", True))

    max_steps = args.steps if args.steps is not None else cfg_steps
    input_temperature = args.temperature if args.temperature is not None else cfg_temperature
    temperature = 0.0 if args.greedy else max(float(input_temperature), 0.0)
    top_k = int(args.top_k) if args.top_k is not None else cfg_top_k
    stop_on_eos = parse_bool_flag(args.stop_on_eos, default=cfg_stop_on_eos)
    do_sample = temperature > 0.0

    messages = _parse_messages(args)
    base_messages = _base_messages_for_new_context(args)
    has_non_system_seed = any(message.get("role") != "system" for message in messages)
    interactive = bool(args.window or args.interactive or not has_non_system_seed)
    checkpoint_path = resolve_checkpoint_path(cfg, args.checkpoint, args.checkpoint_dir)
    if not args.window:
        print(f"Using checkpoint: {checkpoint_path}")

    tokenizer = load_tokenizer(cfg)

    model_context_length = int(cfg.model.context_length)
    requested_context_length = args.max_context or model_context_length
    context_length = requested_context_length

    params = load_params(checkpoint_path)
    rng = jax.random.PRNGKey(args.seed)
    rng, vocab_align_key = jax.random.split(rng)
    tokenizer, params, _, _ = align_tokenizer_and_params_vocab(tokenizer, params, rng_key=vocab_align_key)

    model = build_model(cfg, len(tokenizer), context_length)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    device = jax.devices()[0]
    key_params, key_dropout, sample_key = jax.random.split(rng, 3)
    _, nonparam = init_inference_state(
        model,
        key_params,
        key_dropout,
        batch_size=1,
        pad_token_id=pad_token_id,
        use_kv_cache=True,
    )
    params = jax.device_put(params, device)
    base_state = jax.device_put(nonparam, device)
    prefill_fn, decode_fn = make_prefill_and_decode_fns(model)

    def print_response(response_text: str, prefill_time: float, decode_time: float) -> None:
        print(f"GIANT: {response_text}\n")
        if args.verbose:
            toks_per_s = (max_steps / decode_time) if decode_time > 0 else float("inf")
            print("[perf]")
            print(f"prefill_time_s: {prefill_time:.6f}")
            print(f"decode_time_s:  {decode_time:.6f}")
            print(f"tokens_per_second_decode: {toks_per_s:.6f}\n")

    if args.window:
        _run_window_chat(
            args=args,
            checkpoint_path=checkpoint_path,
            messages=messages,
            base_messages=base_messages,
            has_non_system_seed=has_non_system_seed,
            tokenizer=tokenizer,
            context_length=context_length,
            max_steps=max_steps,
            stop_on_eos=stop_on_eos,
            params=params,
            base_state=base_state,
            prefill_fn=prefill_fn,
            decode_fn=decode_fn,
            temperature=temperature,
            top_k=top_k,
            do_sample=do_sample,
            sample_key=sample_key,
        )
        return

    if messages and has_non_system_seed:
        response_text, prefill_time, decode_time, sample_key = _run_turn(
            tokenizer=tokenizer,
            messages=messages,
            context_length=context_length,
            steps=max_steps,
            stop_on_eos=stop_on_eos,
            params=params,
            base_state=base_state,
            prefill_fn=prefill_fn,
            decode_fn=decode_fn,
            temperature=temperature,
            top_k=top_k,
            do_sample=do_sample,
            sample_key=sample_key,
        )
        print_response(response_text, prefill_time, decode_time)
        messages.append({"role": "assistant", "content": response_text})
        if not interactive:
            return

    print("Enter text to chat with the model. Empty line or Ctrl+D exits.\n")
    while True:
        try:
            user_text = input("User: ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        user_clean = user_text.strip()
        if not user_clean:
            print("Exiting.")
            break

        if user_clean == "/new":
            messages = list(base_messages)
            print("Started new conversation.\n")
            continue

        if user_clean == "/clear":
            print("\033[2J\033[H", end="")
            continue

        messages.append({"role": "user", "content": user_clean})
        response_text, prefill_time, decode_time, sample_key = _run_turn(
            tokenizer=tokenizer,
            messages=messages,
            context_length=context_length,
            steps=max_steps,
            stop_on_eos=stop_on_eos,
            params=params,
            base_state=base_state,
            prefill_fn=prefill_fn,
            decode_fn=decode_fn,
            temperature=temperature,
            top_k=top_k,
            do_sample=do_sample,
            sample_key=sample_key,
        )
        messages.append({"role": "assistant", "content": response_text})
        print_response(response_text, prefill_time, decode_time)


if __name__ == "__main__":
    main()
