import triton
import triton.language as tl
import torch
from zipvllm.kernel.utils import _strides


@triton.jit
def compress_score_out_order_kernel(
    score_cache_ptr,
    save_indices_ptr,
    load_indices_ptr,
    block_table_ptr,
    stride_sb,
    stride_sl,
    stride_sh,
    stride_tb,
    stride_tn,
):
    pid = tl.program_id(0)
    batch_idx = tl.load(save_indices_ptr + pid * 4)
    head_idx = tl.load(save_indices_ptr + pid * 4 + 1)

    save_block_idx = tl.load(save_indices_ptr + pid * 4 + 2)
    save_block_id = tl.load(
        block_table_ptr + batch_idx * stride_tb + save_block_idx * stride_tn
    )
    save_slot_idx = tl.load(save_indices_ptr + pid * 4 + 3)

    save_score_base = (
        score_cache_ptr
        + head_idx * stride_sh
        + save_block_id * stride_sb
        + save_slot_idx * stride_sl
    )

    load_block_idx = tl.load(load_indices_ptr + pid * 4 + 2)
    load_block_id = tl.load(
        block_table_ptr + batch_idx * stride_tb + load_block_idx * stride_tn
    )
    if load_block_id < 0:
        load_block_id = -load_block_id - 2
    load_slot_idx = tl.load(load_indices_ptr + pid * 4 + 3)

    load_score_base = (
        score_cache_ptr
        + head_idx * stride_sh
        + load_block_id * stride_sb
        + load_slot_idx * stride_sl
    )

    s = tl.load(load_score_base)
    tl.store(save_score_base, s)


def compress_score_out_order(
    score_cache: torch.Tensor,
    save_indices: torch.Tensor,
    load_indices: torch.Tensor,
    block_table: torch.Tensor,
):
    """
    compress score

    Arguments:
        score_cache: (num_kvcache_blocks, block_size, num_kv_heads)
        save_indices: (N, 4)
        load_indices: (N, 4)
        block_table: (batch_size, max_num_blocks_per_seq)
    """
    N = save_indices.shape[0]
    grid = (N,)
    compress_score_out_order_kernel[grid](
        score_cache,
        save_indices,
        load_indices,
        block_table,
        **_strides(score_cache, "sb", "sl", "sh"),
        **_strides(block_table, "tb", "tn"),
    )
