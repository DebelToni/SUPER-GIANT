````md
# TiDAR (Think in Diffusion, Talk in Autoregression) — Implementation Notes & Pseudocode
> Goal: extend an existing dense autoregressive LLM training + inference stack to support TiDAR-style **one-pass** decoding:  
> **Talk** = autoregressive (AR) sampling for “current draft” tokens  
> **Think** = diffusion-style (masked, blockwise bidirectional) drafting for “next-step proposals”  
>
> This doc is written so you can hand it to an internal agent to implement TiDAR on top of a normal transformer.

---

## 0) What TiDAR is (high level)
TiDAR keeps **one standard decoder-only transformer** (same layers, same LM head), but trains it to operate under **two attention regimes**:

1) **Causal / AR mode (Talk)**: standard next-token prediction + standard causal self-attention.
2) **Diffusion / masked mode (Think)**: tokens are **[MASK]** inputs; attention is **blockwise bidirectional** (within blocks) with **inter-block causality** (left-to-right across blocks).  
   The diffusion-mode predicts tokens at **aligned positions** (not shifted) from masked inputs.

In training, both modes are exercised in **one forward pass** using a **doubled sequence**:
- First half: clean tokens, causal attention, NTP labels (shifted)
- Second half: masked copy, diffusion attention, diffusion labels (aligned)

In inference, TiDAR constructs a **structured attention mask** so a single forward pass simultaneously:
- samples/verifies the current draft AR-style (Talk),
- and pre-drafts the next proposals in diffusion-style (Think),
using “free token slots” (extra positions) in the same pass.

---

## 1) Key terms and shapes
- `S`: training sequence length (e.g., 128, 2048)
- `K`: diffusion block length / draft length (e.g., 4, 8, 16)
- `L`: current prefix length during inference (tokens already committed)
- `MASK_ID`: vocab id reserved for `[MASK]` (must exist in tokenizer/vocab)
- `q_len`: “draft-part” token count during inference step  
  In TiDAR-style packing this is typically: `q_len = K + K*K` (K verify tokens + K² pre-draft mask tokens)

---

## 2) The one thing your current JAX block MUST gain: flexible masks + position_ids
Your current attention module (from the pasted file):
- `NativeJaxSelfAttention.__call__(x, deterministic, use_kv_cache=False, cur_index=None)`
- uses:
  - `is_causal=True` in non-cache path (hardcoded causal)
  - a simple cache bias in cache path (only masks future keys)
  - RoPE positions assumed as `0..l-1` (or single `cur_index`)

**TiDAR requires:**
1) **Arbitrary attention bias/mask** (not just causal) for both training and inference.
2) **Per-token position_ids** for RoPE (because TiDAR reorders tokens in tensors and also creates multiple “copies” of positions for candidate drafts).

### Minimum API changes
Modify your attention call to accept:
- `attn_bias: Optional[jnp.ndarray]` broadcastable to `(b, heads, q_len, kv_len)` (or `(1,1,q_len,kv_len)`)
- `position_ids: Optional[jnp.ndarray]` of shape `(b, seq_len)` (int32)

Pseudo-signature:
```python
def __call__(self, x, *, deterministic: bool,
             attn_bias: Optional[jnp.ndarray] = None,
             position_ids: Optional[jnp.ndarray] = None,
             use_kv_cache: bool = False,
             cur_index: Optional[int] = None):
    ...
````

### RoPE change (important)

Right now RoPE uses:

* non-cache: `sin = rope_sin[:, :l]`, positions are implicitly `0..l-1`
* cache: `sin = rope_sin[:, cur_index:cur_index+1]`

For TiDAR, you must be able to do:

* `sin = rope_sin[:, position_ids]` (gather per token position)
* same for `cos`

Pseudo:

```python
# position_ids: (b, l)
sin = jnp.take(self._rope_sin[0, :, 0, :], position_ids, axis=0)  # (b, l, rot_dim)
cos = jnp.take(self._rope_cos[0, :, 0, :], position_ids, axis=0)
sin = sin[:, :, None, :]  # (b, l, 1, rot_dim) to broadcast over heads
cos = cos[:, :, None, :]
```

---

## 3) Training: data construction, mask, labels, loss

### 3.1 Training input construction (single forward pass)

Given a clean token sequence:

* `x = [x0, x1, ..., x_{S-1}]` shape `(S,)`

Construct the model input:

* `clean = x`  (length S)
* `diff  = [MASK_ID] * S`  (length S)
* `input_ids = concat(clean, diff)`  (length 2S)

### 3.2 Training labels

You compute **two losses** in the same forward pass.

#### (A) NTP labels for clean half (shifted)

For positions `0..S-2`:

* target at clean position `t` is `x[t+1]`
  Position `S-1` is ignored (or EOS depending on your usual training).

So:

* `labels_ntp[t] = x[t+1]` for `t in [0..S-2]`
* `labels_ntp[S-1] = IGNORE`

#### (B) Diffusion labels for diffusion half (aligned)

For diffusion positions `S..2S-1`:

* diffusion position `S+t` predicts `x[t]`

So:

* `labels_diff[S+t] = x[t]` for `t in [0..S-1]`

### 3.3 Training position_ids

Critical: diffusion predictions are **aligned** to original token positions.

A simple and effective scheme:

* clean half gets `pos = [0..S-1]`
* diffusion half ALSO gets `pos = [0..S-1]` (aligned)

So:

* `position_ids = [0..S-1, 0..S-1]` length `2S`

This is why you need per-token `position_ids`: the diffusion tokens appear after clean tokens in the tensor, but they must behave as if they were at the same absolute positions.

### 3.4 Training attention mask: “Block Diffusion” (non-overlapping blocks)

This is the big conceptual point we discussed:

**Diffusion is NOT trained on sliding windows** like:

* `t1..tK`, `t2..tK+1`, `t3..tK+2`, ...

Instead diffusion uses **non-overlapping blocks**:

* `t0..tK-1`, `tK..t2K-1`, `t2K..t3K-1`, ...

There are about `ceil(S/K)` blocks in the diffusion half.

#### Mask design goals

1. Clean half behaves like a normal AR LM (causal among clean tokens).
2. Diffusion half tokens are `[MASK]` inputs, predicted with:

   * **bidirectional attention within a block**
   * **causal flow across blocks** (block 0 → block 1 → block 2 …)
3. Prevent label leakage: diffusion token for position `t` must **not** be allowed to attend to the clean token `x[t]` (the answer).

#### Practical training mask (conceptual rules)

Let:

* `type[i] in {CLEAN, DIFF}`
* `pos[i]` = position_id at token i (0..S-1)

**Clean queries (CLEAN → keys):**

* allow attending to CLEAN keys with `pos[key] <= pos[q]` (standard causal)
* disallow attending to DIFF keys (simplifies, avoids weird feedback loops)

**Diff queries (DIFF → keys):**
Let `block_id(p) = p // K`, `block_start = block_id(p) * K`.

For a DIFF query at position `p = pos[q]`:

* allow attending to DIFF keys with:

  * same block: `block_id(pos[k]) == block_id(p)` (bidirectional within block)
  * earlier blocks: `block_id(pos[k]) < block_id(p)` (inter-block causal)
* allow attending to CLEAN keys only for positions strictly before the current block:

  * `pos[key] < block_start`
    (this avoids seeing the target tokens in the same block)

This gives the diffusion block enough prefix context while preventing trivial copying.

### 3.5 Loss normalization and the “1:1” weight

Even though diffusion is block-structured, with full-masking you still get **~S diffusion CE terms** (one per diffusion token). NTP gives **~S** terms (S-1).
So a simple loss works well:

* `L = mean(NTP_CE_terms) + mean(DIFF_CE_terms)`
  This is what “a=1, b=1 (1:1)” effectively means: each loss is averaged over its own token set, then summed.

---

## 4) Inference: prefill + iterative decode with Talk+Think in one forward

### 4.1 What happens at inference time (conceptual loop)

Each TiDAR decoding iteration does:

1. **Talk (AR verify/sample)**:

   * You have `K` proposed draft tokens from the previous iteration.
   * You compute logits for these positions and perform rejection sampling to decide:

     * how many tokens to commit this step (`r`),
     * and ensure you advance at least 1 token (standard rejection sampling logic).

2. **Think (diffusion pre-draft)**:

   * In the same forward pass, you also compute `K` candidate next-drafts,
     one for each possible advancement `r` (typically `r in [1..K]`).
   * You pick the candidate corresponding to the realized `r` and use it as the next iteration’s `K` proposed tokens.

This is why inference uses “quadratic slots”: `K²` mask tokens = `K candidate drafts × K tokens each`.

### 4.2 Prefill step (no previous draft yet)

Given a prompt of length `L`:

* You need to produce the first `K` draft tokens for positions `[L..L+K-1]`.

A common TiDAR trick is to build an input tensor like:

* tokens: `[MASK]*K + prompt_tokens`
* position_ids:

  * the `K` masks get positions `[L..L+K-1]`
  * the prompt tokens get positions `[0..L-1]`

Mask:

* prompt tokens: causal among themselves
* mask tokens:

  * can attend to the entire prompt (as prefix)
  * bidirectional among the `K` masks (since you want parallel drafting)

Then take the logits at the mask positions and sample `K` draft tokens.

### 4.3 Decode step (single forward produces verify logits + next proposals)

At decode iteration `t`, you conceptually have:

* prefix tokens length `L` (already committed; normally KV-cached)
* verify/proposed tokens `V` length `K` for positions `[L..L+K-1]`
* pre-draft masks `M` length `K*K` arranged as `K` candidate blocks of size `K`

#### Token layout (conceptual)

Option A (simple to implement first, no reordering):

```
[prefix tokens length L] +
[verify tokens length K] +
[predraft mask tokens length K*K]
```

#### position_ids

* prefix: `0..L-1`
* verify tokens: `L..L+K-1`
* pre-draft candidate blocks:

  * candidate for advance `r` (1..K) predicts positions:
    `[L+r .. L+r+K-1]`  (length K)
  * store these as separate token slots but give them those absolute `position_ids`
  * note: these ranges overlap across candidates; overlap is OK because they’re separate slots.

#### attention mask (core idea)

You want:

* prefix and verify behave like an AR verifier:

  * verify tokens attend to prefix and earlier verify tokens (causal)
  * verify tokens do NOT attend to pre-draft masks (keep AR distribution clean)

For each candidate block `cand[r]`:

* bidirectional within that candidate block
* can attend to prefix and to *the portion of verify tokens that would be “known” under that outcome*

  * simplest conservative rule: candidate r can attend to prefix and to verify tokens positions `<= L+r-1`
  * candidate blocks must not attend to other candidate blocks

#### Rejection sampling + choosing next draft

From verify logits:

* run rejection sampling to get `r` tokens to commit
* select `cand[r]` as the next proposed draft block for the next iteration

Update:

* prefix := prefix + committed_tokens
* L := L + r
* verify := selected cand[r] (K tokens)

Repeat.

---

## 5) Pseudocode

### 5.1 Training step pseudocode

```python
def build_train_batch(x_ids: int[S], K: int):
    # input ids
    clean = x_ids                      # [S]
    diff  = [MASK_ID] * S              # [S]
    input_ids = concat(clean, diff)    # [2S]

    # position ids (aligned)
    pos = arange(S)                    # [S]
    position_ids = concat(pos, pos)    # [2S]

    # labels
    labels = full([2S], IGNORE)
    # NTP on clean half
    labels[0:S-1] = clean[1:S]
    # diffusion aligned on diff half
    labels[S:2S]  = clean[0:S]

    # loss masks (boolean)
    loss_mask_ntp  = zeros([2S]); loss_mask_ntp[0:S-1] = 1
    loss_mask_diff = zeros([2S]); loss_mask_diff[S:2S]  = 1

    # attention bias
    attn_bias = build_tidar_train_bias(position_ids, token_types=[CLEAN]*S+[DIFF]*S, K=K)
    # attn_bias shape: [1, 1, 2S, 2S], float(0 or -inf)

    return input_ids, position_ids, labels, loss_mask_ntp, loss_mask_diff, attn_bias

def train_step(params, batch):
    logits = model(params,
                   input_ids=batch.input_ids,
                   position_ids=batch.position_ids,
                   attn_bias=batch.attn_bias,
                   deterministic=False)

    # CE per token
    ce = cross_entropy(logits, batch.labels)  # [2S], IGNORE handled

    L_ntp  = sum(ce * batch.loss_mask_ntp)  / sum(batch.loss_mask_ntp)
    L_diff = sum(ce * batch.loss_mask_diff) / sum(batch.loss_mask_diff)

    loss = L_ntp + 1.0 * L_diff   # "1:1"
    return loss
```

### 5.2 Inference prefill pseudocode (no KV cache, simplest)

```python
def tidar_prefill(prompt_ids: int[L], K: int):
    # tokens: masks then prompt (implementation convenience)
    input_ids = concat([MASK_ID]*K, prompt_ids)    # [K+L]

    # position ids:
    # masks represent positions L..L+K-1
    pos_masks = arange(L, L+K)      # [K]
    pos_prompt = arange(0, L)       # [L]
    position_ids = concat(pos_masks, pos_prompt)   # [K+L]

    attn_bias = build_tidar_prefill_bias(L=L, K=K, layout="masks_then_prompt")

    logits = model(input_ids, position_ids, attn_bias, deterministic=True)

    # sample draft tokens from mask positions
    draft_logits = logits[0:K]    # positions of the K masks in tensor
    draft_tokens = sample(draft_logits)  # [K], tokens for positions L..L+K-1
    return draft_tokens
```

### 5.3 Inference decode step pseudocode (no cache first)

```python
def tidar_decode_step(prefix_ids: int[L], verify_ids: int[K], K: int):
    # Build pre-draft mask slots: K candidates × K tokens = K*K
    predraft_ids = [MASK_ID] * (K*K)

    input_ids = concat(prefix_ids, verify_ids, predraft_ids)

    # position ids:
    pos_prefix = arange(0, L)
    pos_verify = arange(L, L+K)

    # candidates r=1..K
    pos_predraft = []
    for r in range(1, K+1):
        pos_predraft.extend(arange(L+r, L+r+K))  # length K
    pos_predraft = array(pos_predraft)  # length K*K

    position_ids = concat(pos_prefix, pos_verify, pos_predraft)

    attn_bias = build_tidar_decode_bias(L=L, K=K)  # structured: prefix causal, verify causal, candidates isolated etc.

    logits = model(input_ids, position_ids, attn_bias, deterministic=True)

    # 1) Talk: compute rejection sampling over verify tokens
    verify_logits = logits[L : L+K]   # tensor indices where verify tokens sit
    r, committed_tokens = rejection_sample(prefix_ids, verify_ids, verify_logits)
    # r in [1..K], committed_tokens length r

    # 2) Think: choose candidate r as next verify
    # candidate blocks are laid out sequentially in predraft_ids:
    # candidate r is block index (r-1)
    start = (r-1)*K
    end   = start + K
    cand_logits = logits[L+K + start : L+K + end]
    next_verify = sample(cand_logits)    # [K] tokens for positions L+r..L+r+K-1

    # update prefix
    new_prefix = concat(prefix_ids, committed_tokens)
    return new_prefix, next_verify
```

---

## 6) Mask builders (what your agent should implement)

You need 3 bias builders:

1. `build_tidar_train_bias(position_ids, token_types, K)`
2. `build_tidar_prefill_bias(L, K, layout)`
3. `build_tidar_decode_bias(L, K)`  (structured, K candidates)

Represent masks as **additive bias**:

* allowed: `0.0`
* blocked: `-1e10` (or `-jnp.inf` if kernel supports)

Bias must broadcast to:

* `[1, 1, q_len, kv_len]` for dot_product_attention

### 6.1 Training bias (rule-based, no full O(n²) python loops)

Use broadcasting comparisons on:

* token type (clean/diff)
* aligned `position_ids`
* block ids: `block_id = pos // K`

Then compute:

* `allow_clean_to_clean = (type_q==CLEAN) & (type_k==CLEAN) & (pos_k <= pos_q)`
* `allow_clean_to_diff = False` (optional)
* `allow_diff_to_diff  = (type_q==DIFF) & (type_k==DIFF) & (block_k < block_q OR (block_k==block_q))`

  * plus optionally prevent attending to future blocks: `block_k <= block_q` (causal across blocks)
* `allow_diff_to_clean = (type_q==DIFF) & (type_k==CLEAN) & (pos_k < block_start(pos_q))`

Finally:

* `allow = allow_clean_to_clean OR allow_diff_to_diff OR allow_diff_to_clean`
* `bias = where(allow, 0.0, -1e10)`

### 6.2 Decode bias (core TiDAR inference structure)

You need:

* prefix causal: prefix queries attend to prefix keys (causal)
* verify causal: verify queries attend to prefix + earlier verify (causal)
* candidate blocks:

  * no attention between candidates
  * candidate r queries attend to:

    * prefix
    * verify tokens up to index r-1 (positions <= L+r-1)
    * within its candidate block: bidirectional

This is the hardest mask. Build it using indices and block membership.

---

## 7) Practical implementation plan (incremental)

### Phase 1 — “Correctness first” (no KV cache)

1. Modify your `NativeJaxSelfAttention` to accept:

   * `attn_bias`
   * `position_ids` (RoPE gather)
2. Implement training data construction for doubled sequence.
3. Implement `build_tidar_train_bias`.
4. Train a small model and sanity-check:

   * diffusion loss decreases
   * no leakage: if you intentionally allow diffusion→clean within-block, loss collapses suspiciously fast (bad sign)
5. Implement inference (no cache) for small context lengths:

   * prefill to generate first draft
   * iterative decode step with structured mask and rejection sampling

### Phase 2 — Add KV cache

* Cache only the **prefix (committed tokens)** keys/values.
* Each step, run the model only on the “draft part” tokens (verify + predraft slots) while letting them attend to cached prefix via bias.
* You will need an attention implementation that supports:

  * `keys = concat(k_cache, k_new)`
  * `bias` spanning `[q_len_new, kv_len_total]`
* Your existing cache code already passes a bias; extend it from “mask invalid future positions” to “full structured bias”.

### Phase 3 — Performance work

* Avoid materializing huge `(2S x 2S)` masks for large `S`:

  * compute bias using broadcasting on vectors (still materializes 2D)
  * consider block-sparse attention later if needed
* Compile separate graphs for fixed K values (K is usually fixed per run).
* Keep K modest (4–16) initially.

---

## 8) Notes specific to your current code

You provided:

* `NativeJaxSelfAttention` using `jax.nn.dot_product_attention`
* in non-cache mode it is hard `is_causal=True`
* in cache mode it is `is_causal=False` with a simple bias masking invalid keys
* RoPE cache assumes monotonic positions

TiDAR will require:

* always calling attention with `is_causal=False` and your own `bias` when using TiDAR masks
* a `position_ids`-based RoPE gather path (both training and inference)
* careful separation of:

  * tensor order (for efficiency)
  * absolute positions (via `position_ids`)

---

## 9) Mental model check (the two biggest “gotchas”)

1. **Training diffusion blocks are non-overlapping** (block diffusion), not sliding windows.
   That’s why training can be done in a single forward pass without combinatorial variants.

2. **Aligned diffusion labels** means diffusion token slot with position_id `t` is trained to predict token `x[t]`
   even though the slot appears in a different place in the tensor.

---

If you implement exactly the above (flexible masks + per-token position_ids + doubled training sequence + structured inference mask),
you will have a faithful TiDAR-style extension over a standard dense LLM.
