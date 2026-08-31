from __future__ import annotations

import jax
import jax.numpy as jnp

from TiDAR.model.tidar_core import anchor_rejection_sample_meta


def test_rejection_ratio_sampling_recovers_target_distribution():
    """Proposal sampling plus residual rejection must reproduce target p."""
    target = jnp.array([0.2, 0.8], dtype=jnp.float32)
    proposal = jnp.array([0.8, 0.2], dtype=jnp.float32)
    verify_logits = jnp.stack([jnp.log(target), jnp.log(target)])
    draft_logits = jnp.stack([jnp.log(proposal), jnp.log(proposal)])

    def sample_once(key):
        proposal_key, rejection_key = jax.random.split(key)
        draft_token = jax.random.categorical(proposal_key, jnp.log(proposal)).astype(jnp.int32)
        _, _, committed, _, _ = anchor_rejection_sample_meta(
            rejection_key,
            anchor_token=jnp.asarray(0, dtype=jnp.int32),
            draft_tokens=draft_token[None],
            verify_logits=verify_logits,
            draft_logits=draft_logits,
            temperature=1.0,
            top_k=0,
        )
        return committed[0]

    keys = jax.random.split(jax.random.PRNGKey(0), 100_000)
    samples = jax.jit(jax.vmap(sample_once))(keys)
    empirical = jnp.bincount(samples, length=2) / samples.size
    assert jnp.allclose(empirical, target, atol=0.01)
