import triton
import triton.language as tl
import torch
from zipvllm.kernel.utils import get_padded_headsize, _strides


@triton.jit
def global_score_kernel(
    reduced_scores_ptr,
    score_cache_ptr,
    block_table_ptr,
    compressed_ptr,
    H_kv: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MAX_BLOCKS_PER_SEQ: tl.int32,
    BLOCK_SIZE: tl.constexpr,
    decay_factor: tl.float32,
    activate_method: tl.constexpr,
    stride_sy:tl.int64,
    stride_sb,
    stride_sh,
    stride_slb,
    stride_sp,
    stride_cy:tl.int64,
    stride_cb,
    stride_cp,
    stride_ch,
    stride_tb,
    stride_tn,
):
    pid = tl.program_id(0)
    layer_id = tl.program_id(1)

    batch_idx = pid // (H_kv * MAX_BLOCKS_PER_SEQ)
    rem = pid % (H_kv * MAX_BLOCKS_PER_SEQ)
    kv_head_idx = rem // MAX_BLOCKS_PER_SEQ
    block_idx = rem % MAX_BLOCKS_PER_SEQ
    offs = tl.arange(0, BLOCK_N)
    compressed = tl.load(compressed_ptr + batch_idx)
    block_id = tl.load(block_table_ptr + batch_idx * stride_tb + block_idx * stride_tn)
    if not block_id == -1:
        if block_id < 0:
            block_id = -block_id - 2
            compressed = False
        score_base = (
            reduced_scores_ptr
            + batch_idx * stride_sb
            + kv_head_idx * stride_sh
            + block_idx * stride_slb
            + layer_id * stride_sy
        )
        cache_base = (
            score_cache_ptr
            + block_id * stride_cb
            + kv_head_idx * stride_ch
            + layer_id * stride_cy
        )
        for block_offset in range(0, BLOCK_SIZE, BLOCK_N):
            block_offs = block_offset + offs
            mask = block_offs < BLOCK_SIZE
            reduced_ptr = score_base + block_offs * stride_sp
            cache_ptr = cache_base + block_offs * stride_cp
            reduced_vals = tl.load(reduced_ptr, mask=mask, other=0)
            if compressed:
                cache_vals = tl.load(cache_ptr, mask=mask, other=0)
                if activate_method == "max":
                    out = tl.maximum(reduced_vals, cache_vals * decay_factor)
                elif activate_method == "sum":
                    out = reduced_vals + cache_vals * decay_factor
                out = out.to(reduced_vals.dtype)
            else:
                out = reduced_vals
            tl.store(reduced_ptr, out, mask=mask)
            tl.store(cache_ptr, out, mask=mask)


def global_score(
    reduced_scores: torch.Tensor,
    score_cache: torch.Tensor,
    block_table: torch.Tensor,
    compressed: torch.Tensor,
    decay_factor: float = 0.9,
    activate_method: str = "max",
) -> torch.Tensor:
    """
    calculate max(reduced_scores, score_cache)
    save the result to score_cache, return the result

    Arguments:
        reduced_scores: Shape (num_layers, batch_size, num_kv_heads, max_num_blocks_per_seq, block_size)
        score_cache: Shape (num_layers, num_kvcache_blocks, block_size, num_kv_heads)
        block_table: Shape (batch_size, max_num_blocks_per_seq)
        compressed: Shape (batch_size)
        decay_factor: float
    Returns:
        reduced_scores: Shape (batch_size, num_kv_heads, max_num_blocks_per_seq, block_size)
    """
    BLOCK_N = 256
    assert decay_factor >= 0 and decay_factor <= 1
    assert activate_method in ["max", "sum"]
    num_layers, _, block_size, num_kv_heads = score_cache.shape
    batch_size, max_num_blocks_per_seq = block_table.shape
    grid_store = (batch_size * num_kv_heads * max_num_blocks_per_seq, num_layers)
    global_score_kernel[grid_store](
        reduced_scores,
        score_cache,
        block_table,
        compressed,
        num_kv_heads,
        BLOCK_N,
        max_num_blocks_per_seq,
        block_size,
        decay_factor,
        activate_method,
        **_strides(reduced_scores, "sy", "sb", "sh", "slb", "sp"),
        **_strides(score_cache, "cy", "cb", "cp", "ch"),
        **_strides(block_table, "tb", "tn"),
        num_warps=1,
    )
    return reduced_scores
