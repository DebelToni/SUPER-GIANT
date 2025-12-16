import inspect
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    # Newer Transformers (v4.45+ and v5.x) KV cache interface
    from transformers import DynamicCache
except Exception:
    DynamicCache = None


def _filtered_forward(module, **kwargs):
    """
    Call module.forward(**kwargs) but only pass args that exist in its signature.
    This makes the code resilient across model families (some accept cache_position,
    some don't, etc.).
    """
    sig = inspect.signature(module.forward)
    filt = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return module(**filt)


@torch.no_grad()
def sample_from_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
) -> torch.Tensor:
    """
    logits: (batch, vocab)
    returns: token_ids (batch,)
    """
    if temperature is None or temperature <= 0:
        return logits.argmax(dim=-1)

    logits = logits / float(temperature)

    if top_k and top_k > 0:
        top_k = min(int(top_k), logits.shape[-1])
        vals, idx = torch.topk(logits, top_k, dim=-1)
        masked = torch.full_like(logits, float("-inf"))
        masked.scatter_(dim=-1, index=idx, src=vals)
        logits = masked

    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits.float(), dim=-1)
        cumprobs = torch.cumsum(sorted_probs, dim=-1)

        # mask tokens once cumulative prob exceeds top_p
        sorted_mask = cumprobs > float(top_p)
        sorted_mask[..., 0] = False  # keep at least 1 token

        sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))

        # unsort back to vocab order
        unsorted = torch.full_like(logits, float("-inf"))
        unsorted.scatter_(dim=-1, index=sorted_idx, src=sorted_logits)
        logits = unsorted

    probs = torch.softmax(logits.float(), dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


@torch.no_grad()
def latent_decode(
    model,
    tokenizer,
    prompt: str,
    latent_steps: int = 40,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 0,
    normalize_latents: bool = False,
):
    """
    Latent decoding:
      - Prefill on prompt (input_ids)
      - Then iterate latent_steps times:
          latent_{t+1} = Transformer(latent_t as inputs_embeds for next position)
        where latent_0 is the final hidden state of the prompt's last token.
      - After loop, map all latent vectors through LM head and sample tokens *once*.

    Returns:
      sampled_ids: (1, latent_steps) token ids sampled from final-projected latents
      latents: (1, latent_steps, hidden) latent vectors produced by the loop
    """
    model.eval()

    # Figure out where inputs must live (important when using device_map, etc.)
    emb = model.get_input_embeddings()
    device = emb.weight.device

    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)

    # "Base" transformer module (no LM head)
    # base_model_prefix is usually "model" (Llama/Qwen/etc.) or "transformer" (GPT2)
    base = getattr(model, model.base_model_prefix, None)

    # Prepare cache
    past = DynamicCache(config=model.config) if DynamicCache is not None else None

    # Prefill: process the full prompt into cache
    # cache_position is the absolute index for each token (0..T-1). :contentReference[oaicite:1]{index=1}
    T = input_ids.shape[1]
    cache_position = torch.arange(T, device=device, dtype=torch.long)
    position_ids = cache_position.unsqueeze(0)  # (1, T) works for most decoder-only LMs

    if base is None:
        # Fallback: use the CausalLM wrapper and request hidden states
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
        last_hidden = out.hidden_states[-1]
        past = out.past_key_values
    else:
        out = _filtered_forward(
            base,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past,
            cache_position=cache_position,
            use_cache=True,
            return_dict=True,
        )
        last_hidden = out.last_hidden_state
        past = out.past_key_values

    # latent_0 = final hidden of the last prompt token
    latent = last_hidden[:, -1, :]  # (1, hidden)

    # Optional: try to keep latent scale closer to token embedding scale
    # (this is NOT part of your request, just an option because scales can drift)
    if normalize_latents:
        w = emb.weight
        target_rms = w.float().pow(2).mean(dim=-1).sqrt().mean()
        latent_rms = latent.float().pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        latent = latent * (target_rms / latent_rms).to(latent.dtype)

    latents = []
    cur_pos = T  # next position index

    for _ in range(latent_steps):
        # attention_mask must cover past + current token when using cached forward loops :contentReference[oaicite:2]{index=2}
        attention_mask = torch.cat(
            [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))],
            dim=-1,
        )

        cache_position = torch.tensor([cur_pos], device=device, dtype=torch.long)
        position_ids = cache_position.unsqueeze(0)  # (1, 1)

        # Feed the previous step's *hidden state* directly as the next step's inputs_embeds
        inputs_embeds = latent.unsqueeze(1)  # (1, 1, hidden)

        if base is None:
            out = _filtered_forward(
                model,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past,
                cache_position=cache_position,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            latent = out.hidden_states[-1][:, -1, :]
            past = out.past_key_values
        else:
            out = _filtered_forward(
                base,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past,
                cache_position=cache_position,
                use_cache=True,
                return_dict=True,
            )
            latent = out.last_hidden_state[:, -1, :]
            past = out.past_key_values

        if normalize_latents:
            w = emb.weight
            target_rms = w.float().pow(2).mean(dim=-1).sqrt().mean()
            latent_rms = latent.float().pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
            latent = latent * (target_rms / latent_rms).to(latent.dtype)

        latents.append(latent)
        cur_pos += 1

    latents = torch.stack(latents, dim=1)  # (1, latent_steps, hidden)

    # Project *once at the end* through the LM head (output embedding)
    lm_head = model.get_output_embeddings()
    if lm_head is None:
        raise RuntimeError("Model has no output embeddings / lm_head (get_output_embeddings() returned None).")

    logits = lm_head(latents)  # (1, latent_steps, vocab)

    # Now do softmax + sampling per position (sampling does NOT affect subsequent latents)
    sampled = []
    for t in range(latent_steps):
        token_id = sample_from_logits(
            logits[:, t, :],
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        sampled.append(token_id)

    sampled_ids = torch.stack(sampled, dim=1)  # (1, latent_steps)
    return sampled_ids, latents, logits


def main():
    model_id = "Qwen/Qwen3-0.6B"  # swap with your model (e.g., "meta-llama/Meta-Llama-3-8B-Instruct")
    prompt = "Write a short mystery story opening:\n"

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    # If your tokenizer has no pad token, it's usually safe to set it to eos for decoding utilities
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    sampled_ids, latents, _ = latent_decode(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        latent_steps=40,
        temperature=0.9,
        top_p=0.95,
        top_k=0,
        normalize_latents=False,  # try True if things explode quickly
    )

    print("\n=== Prompt ===")
    print(prompt)

    print("\n=== Latent-decoded tokens (sampled *after* latent rollout) ===")
    print(tokenizer.decode(sampled_ids[0], skip_special_tokens=True))

    # Baseline: normal autoregressive generation for comparison
    print("\n=== Normal generate() baseline (40 new tokens) ===")
    inp = tokenizer(prompt, return_tensors="pt")
    out_ids = model.generate(**inp, max_new_tokens=40, do_sample=True, temperature=0.9, top_p=0.95)
    print(tokenizer.decode(out_ids[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
