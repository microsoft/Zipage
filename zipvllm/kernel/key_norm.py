import triton
import triton.language as tl
import torch
from zipvllm.kernel.utils import _strides


@triton.jit
def norm_kernel(
    key_cache_ptr,
    key_norm_ptr,
    block_table_ptr,
    stride_ky,
    stride_kb,
    stride_kl,
    stride_kh,
    stride_kd,
    stride_ny,
    stride_nz,
    stride_nh,
    stride_nb,
    stride_nl,
    stride_tz,
    stride_tb,
    BLOCK_S: tl.constexpr,
    max_num_blocks_per_seq: tl.int32,
    block_size: tl.constexpr,
    batch_size: tl.int32,
    head_dim: tl.constexpr,
):
    pid = tl.program_id(0)
    layer_id = tl.program_id(1)
    batch_idx = pid % batch_size
    rem = pid // batch_size
    head_idx = rem // max_num_blocks_per_seq
    block_idx = rem % max_num_blocks_per_seq
    block_id = tl.load(block_table_ptr + batch_idx * stride_tz + block_idx * stride_tb)
    if not block_id == -1:
        if block_id < 0:
            block_id = -block_id - 2
        for block_offset in range(0, block_size, BLOCK_S):
            key_ptr = tl.make_block_ptr(
                base=key_cache_ptr
                + block_id * stride_kb
                + head_idx * stride_kh
                + layer_id * stride_ky,
                shape=(block_size, head_dim),
                strides=(stride_kl, stride_kd),
                offsets=(block_offset, 0),
                block_shape=(BLOCK_S, head_dim),
                order=(1, 0),
            )
            key = tl.load(key_ptr, boundary_check=(0,))
            raw_dtype = key.dtype
            key = key.to(tl.float32)
            norm_val = tl.sqrt(tl.sum(key * key, axis=1))
            norm_val = norm_val.to(raw_dtype)
            norm_ptr = tl.make_block_ptr(
                base=key_norm_ptr
                + batch_idx * stride_nz
                + head_idx * stride_nh
                + block_idx * stride_nb
                + layer_id * stride_ny,
                shape=(block_size,),
                strides=(stride_nl,),
                offsets=(block_offset,),
                block_shape=(BLOCK_S,),
                order=(0,),
            )
            tl.store(norm_ptr, norm_val, boundary_check=(0,))


def key_norm(
    key_cache: torch.Tensor,
    block_table: torch.Tensor,
    norm_epsilon: float = 1e-6,
):
    """
    Calculate the norm of the key cache.
    Args:
        key_cache: (num_layers, num_kvcache_blocks, block_size, num_kv_heads, head_dim)
        block_table: (batch_size, max_num_blocks_per_seq)
        last_block: (batch_size)
    Returns:
        key_norm: (num_layers, batch_size, num_kv_heads, max_num_blocks_per_seq , block_size)
    """
    BLOCK_S = 256

    num_layers, _, block_size, num_kv_heads, head_dim = key_cache.shape
    batch_size, max_num_blocks_per_seq = block_table.shape

    key_norm = torch.zeros(
        (num_layers, batch_size, num_kv_heads, max_num_blocks_per_seq, block_size),
        device=key_cache.device,
        dtype=key_cache.dtype,
    )
    grid = (batch_size * num_kv_heads * (max_num_blocks_per_seq), num_layers)

    norm_kernel[grid](
        key_cache,
        key_norm,
        block_table,
        **_strides(key_cache, "ky", "kb", "kl", "kh", "kd"),
        **_strides(key_norm, "ny", "nz", "nh", "nb", "nl"),
        **_strides(block_table, "tz", "tb"),
        max_num_blocks_per_seq=max_num_blocks_per_seq,
        BLOCK_S=BLOCK_S,
        block_size=block_size,
        batch_size=batch_size,
        head_dim=head_dim,
    )
    key_norm = torch.clamp(key_norm, min=norm_epsilon)
    return key_norm
