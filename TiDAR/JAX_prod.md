In “naïve” JAX generation you’d feed `input_ids[:, :t]` each step, so **the sequence dimension changes** and you’d trigger a **new XLA compile** for every new `t`. Production JAX stacks avoid that almost entirely by making the *physical* KV-cache shape static and letting only a small **scalar index** grow.

## What production JAX stacks actually do

### 1) Preallocate a fixed-size KV cache, grow a *cursor* (no recompiles per token)

The canonical Flax/JAX pattern is:

* Allocate `cached_key` / `cached_value` with a **fixed `max_length`** dimension.
* Keep an integer `cache_index` that starts at 0.
* Each decode step writes the new token’s K/V into the cache at `cache_index` using `lax.dynamic_update_slice`.
* Build a mask from `cache_index` so attention can’t see the “future” (still-unwritten) part of the cache.

You can see this directly in Flax’s `MultiHeadDotProductAttention(decode=True)` implementation: it updates the cache via `lax.dynamic_update_slice`, increments `cache_index`, and creates a mask with `jnp.arange(max_length) <= cur_index`. ([flax.readthedocs.io][1])

**Why this works with JAX static shapes:** the KV arrays are always `[batch…, max_length, heads, head_dim]` (static), and only `cache_index` changes (a scalar), so the compiled program stays valid.

---

### 2) Bucketed compilation for *prompt length* and *batch size* (compile a small set of shapes)

Even with a fixed `max_length`, you don’t want *one* gigantic shape for everything (e.g., always `max_length=128k`, always `batch=256`) because that hurts memory and can reduce efficiency.

What engines do instead is **compile a small menu of shapes** and route requests:

* **Context-length buckets** (e.g., 512/1k/2k/4k/8k/16k…): pad prompts up to the bucket and use that compiled prefill.
* **Batch-size buckets** (e.g., 1/2/4/8/16/32…): pad the “active slots” to a supported batch size and mask inactive slots.

This is explicitly described in the “continuous batching” inference-engine design: compile **multiple prefill functions** (for variable context lengths / limits) and a **generate function** that runs on the current active batch. ([jax-ml.github.io][2])

A concrete production example: SGLang-Jax states its scheduler **pre-compiles JAX computation graphs for different batch sizes**. ([lmsys.org][3])

And for Google’s TPU serving stack, JetStream is documented as providing **continuous batching** and **KV cache optimizations** as part of a production-oriented serving framework. ([Google Cloud Documentation][4])

---

### 3) Paged + ragged attention to avoid “one dense `[max_length]` per sequence”

When you push to long context / high concurrency, the *memory layout* of KV cache becomes the bottleneck (fragmentation, waste, variable lengths). Modern TPU/JAX serving stacks increasingly use **paged attention**:

* KV cache is stored in **fixed-size pages** (e.g., 128 tokens per page).
* Each sequence has a **page table** mapping logical positions → physical pages.
* Attention kernels “gather” only the pages that exist for each sequence (“ragged”), instead of assuming a single contiguous dense `[max_length]` buffer.

MaxText’s docs call out “Serving Attention (Paged & Ragged)” as a key inference optimization, explicitly describing fetching non-contiguous KV-cache “pages” and handling ragged layouts. ([maxtext.readthedocs.io][5])

SGLang-Jax also highlights integrating “Ragged Paged Attention v3” and exposes page sizing and chunked prefill knobs (e.g. `--page-size=128`, `--chunked-prefill-size=2048`)—these are exactly the sorts of controls you add when you’re managing KV memory in pages rather than dense tensors. ([lmsys.org][3])

**Key point for your question:** paging keeps the *kernel shapes* mostly static (page size, head dims, etc.) while supporting variable per-request lengths via metadata (page tables, lengths), which is much more JAX/XLA-friendly than changing the K/V tensor’s sequence dimension every step.

---

### 4) Separate prefill and decode (“disaggregated serving”) so decode shapes stay stable

Many production systems split:

* **Prefill**: compute KV for the prompt (often large, variable-length).
* **Decode**: repeatedly append 1 token and update KV (shape-stable loop).

The JAX scaling-book notes this explicitly: “Prefill servers generate KV caches that get sent … to the generate servers,” calling it “disaggregated serving.” ([jax-ml.github.io][2])

This separation makes it easier to:

* bucket/compile prefill shapes,
* keep decode as a stable “one-token step” program,
* and do continuous batching on the decode side.

---

### 5) Don’t forget compilation caching in production

Even with bucketing, you’ll compile *some* set of shapes. In production you typically enable a disk compilation cache so restarts don’t recompile everything. JAX documents a **persistent compilation cache** for compiled programs. ([JAX Documentation][6])
SGLang-Jax’s launch snippet also shows setting `jax_COMPILATION_CACHE_DIR` in practice. ([lmsys.org][3])

---

## So, which of your options matches production?

* **Compile for full context length**: *Sometimes*, but usually only **within buckets** (or with paging so kernels don’t explode with max context).
* **Recompile per step**: basically **never** in serious inference.
* **Recompile every N tokens**: not the common pattern; instead engines do **bucketed max lengths** (and/or **paged KV**). You might “migrate” a sequence to a bigger bucket occasionally (copy KV into a larger cache), but with paged KV you usually just allocate more pages rather than recompiling.

## Minimal mental model (what you’d implement)

Here’s the shape-stable approach Flax demonstrates:

```python
# Shapes are fixed at compile time:
# cached_k: [B, MAX_L, H, D], cached_v: [B, MAX_L, H, D]
# cache_index: scalar int32

def decode_step(query_token, cached_k, cached_v, cache_index):
    k_new, v_new = project_kv(query_token)     # [B, 1, H, D]
    cached_k = lax.dynamic_update_slice(cached_k, k_new, (0, cache_index, 0, 0))
    cached_v = lax.dynamic_update_slice(cached_v, v_new, (0, cache_index, 0, 0))

    # prevent attending to unwritten positions:
    mask = (jnp.arange(MAX_L) <= cache_index)[None, None, None, :]  # broadcast
    out = attention(q=query_token, k=cached_k, v=cached_v, mask=mask)

    return out, cached_k, cached_v, cache_index + 1
```

That keeps shapes static and avoids recompilation, while still behaving like the cache “grows”.

---

[1]: https://flax.readthedocs.io/en/v0.6.10/_modules/flax/linen/attention.html "flax.linen.attention"
[2]: https://jax-ml.github.io/scaling-book/inference/ " All About Transformer Inference | How To Scale Your Model "
[3]: https://lmsys.org/blog/2025-10-29-sglang-jax/ "SGLang-Jax: An Open-Source Solution for Native TPU Inference | LMSYS Org"
[4]: https://docs.cloud.google.com/kubernetes-engine/docs/tutorials/serve-multihost-tpu-jetstream "Serve LLMs using multi-host TPUs on GKE with JetStream and Pathways  |  GKE AI/ML  |  Google Cloud Documentation"
[5]: https://maxtext.readthedocs.io/en/latest/guides/optimization/pallas_kernels_performance.html "Optimizing with Pallas kernels — MaxText  documentation"
[6]: https://docs.jax.dev/en/latest/persistent_compilation_cache.html?utm_source=chatgpt.com "Persistent compilation cache"

