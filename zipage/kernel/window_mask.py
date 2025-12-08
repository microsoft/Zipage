import triton
import triton.language as tl
import torch
from zipage.kernel.utils import _strides


@triton.jit
def window_mask_kernel(
    scores_ptr,
    block_table_ptr,
    query_cache_len: tl.constexpr,
    block_size: tl.int32,
    max_num_blocks_per_seq: tl.int32,
    num_kv_heads: tl.int32,
    stride_sy,
    stride_sb,
    stride_sh,
    stride_slb,
    stride_sp,
    stride_tb,
    stride_tn,
):

    pid = tl.program_id(0)
    layer_id = tl.program_id(1)
    batch_idx = pid // num_kv_heads
    kv_head_idx = pid % num_kv_heads
    for block_idx in range(max_num_blocks_per_seq):
        block_id = tl.load(
            block_table_ptr + batch_idx * stride_tb + block_idx * stride_tn
        )
        if not block_id == -1:
            if block_id < 0:
                block_id = -block_id - 2
                offset = block_size - query_cache_len + tl.arange(0, query_cache_len)
                score_base = (
                    scores_ptr
                    + batch_idx * stride_sb
                    + kv_head_idx * stride_sh
                    + block_idx * stride_slb
                    + layer_id * stride_sy
                )
                score_ptr = score_base + offset * stride_sp
                tl.store(score_ptr, float("inf"))


def window_mask(scores: torch.Tensor, block_table: torch.Tensor, query_cache_len: int):
    """
    set the scores of the last query_cache_len tokens to float("inf")
    Arguments:
        scores: Shape (num_layers, batch_size, num_kv_heads, max_num_blocks_per_seq, block_size)
        block_table: Shape (batch_size, max_num_blocks_per_seq)
        query_cache_len: int
    Returns:
        scores: Shape (num_layers, batch_size, num_kv_heads, max_num_blocks_per_seq, block_size)
    """
    num_layers, batch_size, num_kv_heads, max_num_blocks_per_seq, block_size = (
        scores.shape
    )
    grid_window_mask = (batch_size * num_kv_heads, num_layers)
    window_mask_kernel[grid_window_mask](
        scores,
        block_table,
        query_cache_len,
        block_size,
        max_num_blocks_per_seq,
        num_kv_heads,
        **_strides(scores, "sy", "sb", "sh", "slb", "sp"),
        **_strides(block_table, "tb", "tn"),
        num_warps=1,
    )
    return scores
