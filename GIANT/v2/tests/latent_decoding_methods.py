"""
latent_decoding_methods.py

Demonstrates four decoding methods for decoder-only HF Transformers CausalLMs:

1) Normal (discrete) autoregressive:
   vector -> lm_head -> softmax -> sample/argmax -> token_id -> next iteration

2) Soft-embedding feedback (continuous, but still uses lm_head+softmax each step):
   vector -> lm_head -> softmax -> expected_input_embedding -> next iteration

3) Hard-embedding feedback (continuous, uses input/output embeddings but no softmax):
   vector -> lm_head -> argmax -> input_embedding[token_id] -> next iteration

4) Pure latent feedback (continuous, no lm_head inside the loop):
   vector -> next iteration (feed hidden state directly as inputs_embeds)
   then project all produced latents through lm_head at the end to see implied tokens.

Usage:
  python latent_decoding_methods.py --model gpt2 --prompt "Hello" --steps 40

Notes:
- Method (4) is highly out-of-distribution for most models and often collapses.
- Method (2) tends to be much more stable because it stays in the *input embedding space*.
"""

import argparse
import inspect
import os
import random
from dataclasses import dataclass
from typing import Optional, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    # Newer Transformers releases
    from transformers import DynamicCache
except Exception:
    DynamicCache = None


# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _filtered_forward(module, **kwargs):
    """
    Call module.forward(**kwargs) but only pass args that exist in its signature.
    This makes the code resilient across model families / Transformers versions.
    """
    sig = inspect.signature(module.forward)
    params = sig.parameters

    # If forward has **kwargs, keep all arguments (so flags like output_hidden_states
    # are not dropped on models that only declare **kwargs for them).
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return module(**kwargs)

    filt = {k: v for k, v in kwargs.items() if k in params}
    return module(**filt)


@torch.no_grad()
def sample_from_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    do_sample: bool = True,
) -> torch.Tensor:
    """
    logits: (batch, vocab)
    returns token_ids: (batch,)
    """
    if (not do_sample) or (temperature is None) or (temperature <= 0):
        return logits.argmax(dim=-1)

    logits = logits / float(temperature)

    if top_k and top_k > 0:
        k = min(int(top_k), logits.shape[-1])
        vals, idx = torch.topk(logits, k, dim=-1)
        masked = torch.full_like(logits, float("-inf"))
        masked.scatter_(dim=-1, index=idx, src=vals)
        logits = masked

    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits.float(), dim=-1)
        cum = torch.cumsum(sorted_probs, dim=-1)

        sorted_mask = cum > float(top_p)
        sorted_mask[..., 0] = False  # keep at least 1 token
        sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))

        # unsort back
        unsorted = torch.full_like(logits, float("-inf"))
        unsorted.scatter_(dim=-1, index=sorted_idx, src=sorted_logits)
        logits = unsorted

    probs = torch.softmax(logits.float(), dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


@torch.no_grad()
def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    logits: (batch, vocab) -> entropy: (batch,)
    """
    p = torch.softmax(logits.float(), dim=-1).clamp_min(1e-12)
    return -(p * torch.log(p)).sum(dim=-1)


@torch.no_grad()
def rms(x: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    return x.float().pow(2).mean(dim=dim, keepdim=keepdim).sqrt()


# -----------------------------
# Forward helpers
# -----------------------------
@dataclass
class PrefillState:
    device: torch.device
    attention_mask: torch.Tensor      # (1, T)
    past_key_values: Optional[object] # cache object or tuple
    cur_pos: int                      # next position index
    last_hidden: torch.Tensor         # (1, hidden)
    next_logits: torch.Tensor         # (1, vocab)


@torch.no_grad()
def prefill(
    model,
    tokenizer,
    prompt: str,
) -> PrefillState:
    """
    Runs the prompt through the model to initialize KV cache and returns:
      - last_hidden: final-layer hidden state at last prompt token
      - next_logits: logits at last position (i.e., prediction for next token)
      - past_key_values cache
      - attention_mask aligned with prompt
    """
    emb = model.get_input_embeddings()
    device = emb.weight.device

    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)

    # New cache API if available; otherwise let model produce tuple caches
    past = DynamicCache(config=model.config) if DynamicCache is not None else None

    T = input_ids.shape[1]
    cache_position = torch.arange(T, device=device, dtype=torch.long)
    position_ids = cache_position.unsqueeze(0)  # (1, T)

    out = _filtered_forward(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past,
        cache_position=cache_position,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    )

    last_hidden = out.hidden_states[-1][:, -1, :]     # (1, hidden)
    next_logits = out.logits[:, -1, :]                # (1, vocab)
    past = out.past_key_values

    return PrefillState(
        device=device,
        attention_mask=attention_mask,
        past_key_values=past,
        cur_pos=T,
        last_hidden=last_hidden,
        next_logits=next_logits,
    )


@torch.no_grad()
def step_with_input_ids(
    model,
    token_id: torch.Tensor,           # (1,) or (batch,)
    attention_mask: torch.Tensor,     # (1, L)
    past_key_values,
    cur_pos: int,
):
    """
    One cached step using discrete token_id.
    Returns: (attention_mask, last_hidden, next_logits, past_key_values)
    """
    device = attention_mask.device
    attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)

    cache_position = torch.tensor([cur_pos], device=device, dtype=torch.long)
    position_ids = cache_position.unsqueeze(0)  # (1, 1)

    out = _filtered_forward(
        model,
        input_ids=token_id.view(1, 1),
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        cache_position=cache_position,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    )

    last_hidden = out.hidden_states[-1][:, -1, :]   # (1, hidden)
    next_logits = out.logits[:, -1, :]              # (1, vocab)
    past = out.past_key_values
    return attention_mask, last_hidden, next_logits, past


@torch.no_grad()
def step_with_inputs_embeds(
    model,
    embed_vec: torch.Tensor,          # (1, hidden)
    attention_mask: torch.Tensor,     # (1, L)
    past_key_values,
    cur_pos: int,
):
    """
    One cached step using continuous inputs_embeds (embed_vec).
    Returns: (attention_mask, last_hidden, next_logits, past_key_values)
    """
    device = attention_mask.device
    attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)

    cache_position = torch.tensor([cur_pos], device=device, dtype=torch.long)
    position_ids = cache_position.unsqueeze(0)  # (1, 1)

    out = _filtered_forward(
        model,
        inputs_embeds=embed_vec.view(1, 1, -1),
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        cache_position=cache_position,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    )

    last_hidden = out.hidden_states[-1][:, -1, :]
    next_logits = out.logits[:, -1, :]
    past = out.past_key_values
    return attention_mask, last_hidden, next_logits, past


# -----------------------------
# Method 1: normal discrete AR
# -----------------------------
@torch.no_grad()
def decode_method_1_normal(
    model,
    tokenizer,
    prompt: str,
    steps: int,
    temperature: float,
    top_p: float,
    top_k: int,
    do_sample: bool,
):
    st = prefill(model, tokenizer, prompt)

    generated: List[int] = []
    entropies: List[float] = []

    attn = st.attention_mask
    past = st.past_key_values
    cur_pos = st.cur_pos

    logits = st.next_logits
    for _ in range(steps):
        entropies.append(float(entropy_from_logits(logits)[0].cpu()))
        tok = sample_from_logits(logits, temperature=temperature, top_p=top_p, top_k=top_k, do_sample=do_sample)
        generated.append(int(tok[0].cpu()))

        attn, last_h, logits, past = step_with_input_ids(
            model=model,
            token_id=tok.to(st.device),
            attention_mask=attn,
            past_key_values=past,
            cur_pos=cur_pos,
        )
        cur_pos += 1

    return torch.tensor(generated, dtype=torch.long).view(1, -1), entropies


# -----------------------------
# Method 2: soft embedding feedback
# -----------------------------
@torch.no_grad()
def decode_method_2_soft_embedding(
    model,
    tokenizer,
    prompt: str,
    steps: int,
    temperature: float,
    top_p: float,
    top_k: int,
):
    """
    Each step:
      logits = lm_head(h_t)  (via model outputs)
      p = softmax(logits / temperature)
      p' = top-p/top-k truncate (optional)
      e = p' @ E_in
      feed e as inputs_embeds into next iteration

    We also record an "implied token" each step via argmax(logits) for readability,
    but token ids are NOT fed back.
    """
    st = prefill(model, tokenizer, prompt)

    emb = model.get_input_embeddings()
    device = st.device
    E = emb.weight  # (vocab, hidden)

    implied_ids: List[int] = []
    entropies: List[float] = []

    attn = st.attention_mask
    past = st.past_key_values
    cur_pos = st.cur_pos

    logits = st.next_logits
    for _ in range(steps):
        entropies.append(float(entropy_from_logits(logits)[0].cpu()))
        implied_ids.append(int(logits.argmax(dim=-1)[0].cpu()))

        # temperature
        l = logits
        if temperature is not None and temperature > 0:
            l = l / float(temperature)

        probs = torch.softmax(l.float(), dim=-1)  # (1, vocab)

        # top-k on probs
        if top_k and top_k > 0:
            k = min(int(top_k), probs.shape[-1])
            vals, idx = torch.topk(probs, k, dim=-1)
            p2 = torch.zeros_like(probs).scatter_(dim=-1, index=idx, src=vals)
            probs = p2 / p2.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        # top-p on probs
        if top_p is not None and top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
            cum = torch.cumsum(sorted_probs, dim=-1)
            mask = cum > float(top_p)
            mask[..., 0] = False
            sorted_probs = sorted_probs.masked_fill(mask, 0.0)
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            probs = torch.zeros_like(probs).scatter_(dim=-1, index=sorted_idx, src=sorted_probs)

        # expected embedding in input-embedding space
        e = probs.to(E.dtype) @ E  # (1, hidden)

        attn, last_h, logits, past = step_with_inputs_embeds(
            model=model,
            embed_vec=e.to(device),
            attention_mask=attn,
            past_key_values=past,
            cur_pos=cur_pos,
        )
        cur_pos += 1

    return torch.tensor(implied_ids, dtype=torch.long).view(1, -1), entropies


# -----------------------------
# Method 3: hard-embedding feedback (argmax -> embedding; no softmax)
# -----------------------------
@torch.no_grad()
def decode_method_3_hard_embedding(
    model,
    tokenizer,
    prompt: str,
    steps: int,
):
    """
    Each step:
      logits = lm_head(h_t)
      tok = argmax(logits)  (no softmax/top-p/top-k)
      e = input_embedding[tok]
      feed e as inputs_embeds into next iteration

    Returns implied token ids (argmax each step) but feeds embeddings, not ids.
    """
    st = prefill(model, tokenizer, prompt)

    emb = model.get_input_embeddings()
    device = st.device
    E = emb.weight

    implied_ids: List[int] = []
    entropies: List[float] = []

    attn = st.attention_mask
    past = st.past_key_values
    cur_pos = st.cur_pos

    logits = st.next_logits
    for _ in range(steps):
        entropies.append(float(entropy_from_logits(logits)[0].cpu()))
        tok = logits.argmax(dim=-1)  # (1,)
        implied_ids.append(int(tok[0].cpu()))

        e = E[tok].to(device)  # (1, hidden)

        attn, last_h, logits, past = step_with_inputs_embeds(
            model=model,
            embed_vec=e,
            attention_mask=attn,
            past_key_values=past,
            cur_pos=cur_pos,
        )
        cur_pos += 1

    return torch.tensor(implied_ids, dtype=torch.long).view(1, -1), entropies


# -----------------------------
# Method 4: pure latent feedback
# -----------------------------
@torch.no_grad()
def decode_method_4_pure_latent(
    model,
    tokenizer,
    prompt: str,
    steps: int,
    posthoc_temperature: float,
    posthoc_top_p: float,
    posthoc_top_k: int,
    posthoc_do_sample: bool,
    normalize_latents: bool = False,
):
    """
    Loop:
      latent_0 = final hidden state of prompt's last token
      for t in 1..steps:
        feed latent_{t-1} directly as inputs_embeds
        latent_t = new final hidden state

    After the loop:
      logits_t = lm_head(latent_t)
      sample/argmax tokens from logits_t post-hoc (does not affect latent rollout)
    """
    st = prefill(model, tokenizer, prompt)

    emb = model.get_input_embeddings()
    device = st.device
    E = emb.weight

    attn = st.attention_mask
    past = st.past_key_values
    cur_pos = st.cur_pos

    latent = st.last_hidden  # (1, hidden)

    def _normalize(x: torch.Tensor) -> torch.Tensor:
        # Match RMS to typical input embedding RMS (cheap stabilizer)
        target = rms(E, dim=-1).mean()  # scalar
        r = rms(x, dim=-1, keepdim=True).clamp_min(1e-6)
        return x * (target / r).to(x.dtype)

    if normalize_latents:
        latent = _normalize(latent)

    latents: List[torch.Tensor] = []
    latent_rms: List[float] = []

    for _ in range(steps):
        attn, last_h, next_logits, past = step_with_inputs_embeds(
            model=model,
            embed_vec=latent.to(device),
            attention_mask=attn,
            past_key_values=past,
            cur_pos=cur_pos,
        )
        cur_pos += 1

        latent = last_h
        if normalize_latents:
            latent = _normalize(latent)

        latents.append(latent)
        latent_rms.append(float(rms(latent)[0].cpu()))

    latents_seq = torch.stack(latents, dim=1)  # (1, steps, hidden)
    lm_head = model.get_output_embeddings()
    if lm_head is None:
        raise RuntimeError("Model has no output embeddings / lm_head.")

    logits_seq = lm_head(latents_seq)  # (1, steps, vocab)

    # post-hoc decode
    ids = []
    entropies = []
    for t in range(steps):
        logits = logits_seq[:, t, :]
        entropies.append(float(entropy_from_logits(logits)[0].cpu()))
        tok = sample_from_logits(
            logits,
            temperature=posthoc_temperature,
            top_p=posthoc_top_p,
            top_k=posthoc_top_k,
            do_sample=posthoc_do_sample,
        )
        ids.append(int(tok[0].cpu()))

    return torch.tensor(ids, dtype=torch.long).view(1, -1), entropies, latent_rms


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="gpt2")
    ap.add_argument("--prompt", type=str, default="Write a short mystery story opening:\n")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)

    # sampling params (method 1 + posthoc for method 4)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--no_sample", action="store_true", help="Use argmax instead of sampling")

    # method 4 option
    ap.add_argument("--normalize_latents", action="store_true", help="RMS-normalize latent vectors each step (stabilizer)")

    args = ap.parse_args()
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model)

    # Simple placement; for sharded models load with device_map="auto" instead.
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    model.eval()

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    do_sample = not args.no_sample

    print("=" * 90)
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Steps: {args.steps}")
    print(f"Sampling: {'sample' if do_sample else 'argmax'} | temp={args.temperature} top_p={args.top_p} top_k={args.top_k}")
    print("=" * 90)
    print("PROMPT:")
    print(args.prompt)
    print("=" * 90)

    # Method 1
    ids1, H1 = decode_method_1_normal(
        model, tokenizer, args.prompt, args.steps,
        temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, do_sample=do_sample
    )
    text1 = tokenizer.decode(ids1[0], skip_special_tokens=True)

    # Method 2
    ids2, H2 = decode_method_2_soft_embedding(
        model, tokenizer, args.prompt, args.steps,
        temperature=args.temperature, top_p=args.top_p, top_k=args.top_k
    )
    text2 = tokenizer.decode(ids2[0], skip_special_tokens=True)

    # Method 3
    ids3, H3 = decode_method_3_hard_embedding(
        model, tokenizer, args.prompt, args.steps,
    )
    text3 = tokenizer.decode(ids3[0], skip_special_tokens=True)

    # Method 4
    ids4, H4, latent_rms = decode_method_4_pure_latent(
        model, tokenizer, args.prompt, args.steps,
        posthoc_temperature=args.temperature, posthoc_top_p=args.top_p, posthoc_top_k=args.top_k,
        posthoc_do_sample=do_sample,
        normalize_latents=args.normalize_latents
    )
    text4 = tokenizer.decode(ids4[0], skip_special_tokens=True)

    def summarize_entropy(H):
        if not H:
            return "n/a"
        return f"mean={sum(H)/len(H):.3f}, min={min(H):.3f}, max={max(H):.3f}"

    print("\n[1] Normal discrete AR (token feedback)")
    print("-" * 60)
    print(text1)
    print(f"(logit entropy stats) {summarize_entropy(H1)}")

    print("\n[2] Soft-embedding feedback (lm_head+softmax each step; no discrete token feedback)")
    print("-" * 60)
    print(text2)
    print(f"(logit entropy stats) {summarize_entropy(H2)}")

    print("\n[3] Hard-embedding feedback (argmax embedding; no softmax)")
    print("-" * 60)
    print(text3)
    print(f"(logit entropy stats) {summarize_entropy(H3)}")

    print("\n[4] Pure latent feedback (no lm_head inside loop; post-hoc projection)")
    print("-" * 60)
    print(text4)
    print(f"(logit entropy stats, post-hoc) {summarize_entropy(H4)}")
    if latent_rms:
        print(f"(latent RMS) mean={sum(latent_rms)/len(latent_rms):.3f}, min={min(latent_rms):.3f}, max={max(latent_rms):.3f}")
        if args.normalize_latents:
            print("(latent normalization enabled)")

    print("\nDone.")


if __name__ == "__main__":
    main()
