import jax.numpy as jnp

def tidar_structured_mask(prefix_len: int,
                          block_len: int,
                          num_predraft_blocks: int):
    """
    Build a boolean [T, T] attention mask with:
      - prefix + sample: causal
      - each predraft block: can attend to (prefix) + (first k sample tokens) + (its own block bidirectionally)
        where k == block_id

    Token layout:
      [0 .. P-1]                 : prefix (clean, causal)
      [P .. P+B-1]               : sample tokens (clean, causal)
      [P+B .. P+B+N*B-1]         : predraft blocks, N blocks of size B
    """
    P = prefix_len
    B = block_len
    N = num_predraft_blocks

    T = P + B + N * B
    pos = jnp.arange(T)

    is_prefix = pos < P
    is_sample = (pos >= P) & (pos < P + B)
    is_predraft = pos >= (P + B)

    # Indices for sample tokens: 0..B-1 (meaningful only where is_sample)
    sample_idx = pos - P

    # Predraft block id: 0..N-1 (meaningful only where is_predraft)
    pred_off = pos - (P + B)
    pred_block = pred_off // B  # integer block id

    # Broadcast for pairwise rules
    q_pos = pos[:, None]
    k_pos = pos[None, :]

    q_is_prefix = is_prefix[:, None]
    q_is_sample = is_sample[:, None]
    q_is_predraft = is_predraft[:, None]

    k_is_prefix = is_prefix[None, :]
    k_is_sample = is_sample[None, :]
    k_is_predraft = is_predraft[None, :]

    # ---- 1) Causal part: prefix+sample queries attend causally to prefix+sample keys
    causal_keys = k_is_prefix | k_is_sample
    causal_triangle = (q_pos >= k_pos)
    causal_queries = q_is_prefix | q_is_sample
    allowed = causal_queries & causal_keys & causal_triangle

    # ---- 2) Predraft queries: conditioning + within-block bidirectional
    # (a) Always see full prefix
    allowed |= q_is_predraft & k_is_prefix

    # (b) See the first k sample tokens, where k == block_id for that predraft query
    q_block = pred_block[:, None]          # [T,1]
    k_sample_idx = sample_idx[None, :]     # [1,T]
    allowed |= q_is_predraft & k_is_sample & (k_sample_idx < q_block)

    # (c) Bidirectional inside the same predraft block
    k_block = pred_block[None, :]          # [1,T]
    allowed |= q_is_predraft & k_is_predraft & (k_block == q_block)

    return allowed  # [T, T] bool


print("Example mask (P=2, B=4, N=3):")
print("" + "="*30)
mask = tidar_structured_mask(prefix_len=2, block_len=4, num_predraft_blocks=3)
for i in range (mask.shape[0]):
    row = "".join(['# ' if x else '. ' for x in mask[i]])
    print(row)
print("" + "="*30)
