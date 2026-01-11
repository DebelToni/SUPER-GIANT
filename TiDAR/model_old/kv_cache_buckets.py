"""KV cache bucket helpers for efficient inference."""
from typing import Tuple


def select_kv_bucket(required_len: int, max_context: int) -> int:
    """
    Select appropriate KV cache bucket size following production JAX patterns.
    
    Uses power-of-2 bucketing with some intermediate sizes:
    - Small: 128, 256, 512, 1024
    - Medium: 2048, 4096
    - Large: 8192+
    
    Args:
        required_len: Minimum required cache length
        max_context: Maximum context from model config
    
    Returns:
        Bucket size to use (>= required_len, <= max_context)
    """
    # Power-of-2 buckets with some intermediate steps
    buckets = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    
    # Filter to max_context
    valid_buckets = [b for b in buckets if b <= max_context]
    
    # If max_context is not in list, add it
    if max_context not in valid_buckets:
        valid_buckets.append(max_context)
        valid_buckets.sort()
    
    # Find smallest bucket >= required_len
    for bucket in valid_buckets:
        if bucket >= required_len:
            return bucket
    
    # If required exceeds all buckets, use max_context
    return max_context


def calculate_tidar_cache_requirement(
    prompt_len: int,
    num_new_tokens: int,
    draft_len: int,
) -> int:
    """
    Calculate minimum KV cache needed for TiDAR inference.
    
    TiDAR decode step needs cache for:
    - prefix (committed tokens)
    - verify tokens (current draft being verified)
    
    The predraft tokens don't write to cache.
    
    Args:
        prompt_len: Initial prompt length
        num_new_tokens: Target number of new tokens to generate
        draft_len: TiDAR draft length (K)
    
    Returns:
        Minimum cache length needed
    """
    # Worst case: all tokens accepted every step
    # Best case: 1 token per step (all rejected)
    # Average case: ~K/2 tokens per step
    # Use worst case for safety
    max_prefix = prompt_len + num_new_tokens + draft_len
    return max_prefix


def select_tidar_kv_bucket(
    prompt_len: int,
    num_new_tokens: int,
    draft_len: int,
    max_context: int,
) -> Tuple[int, int]:
    """
    Select KV cache bucket for TiDAR inference.
    
    Returns:
        (bucket_size, effective_window) where:
        - bucket_size: KV cache allocation size
        - effective_window: Max tokens that can be generated with this bucket
    """
    required = calculate_tidar_cache_requirement(prompt_len, num_new_tokens, draft_len)
    bucket = select_kv_bucket(required, max_context)
    
    # Calculate how many tokens can actually be generated with this bucket
    effective_window = bucket - prompt_len - draft_len
    
    return bucket, max(0, effective_window)
