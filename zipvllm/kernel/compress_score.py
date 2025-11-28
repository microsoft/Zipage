import triton
import triton.language as tl
import torch
from zipvllm.kernel.utils import _strides


@triton.jit
def compress_score_kernel(
    score_cache_ptr,
    flag_ptr,
    block_table_ptr,
    target_block_table_ptr,
    stride_sb,
    stride_sl,
    stride_sh,
    stride_fz,
    stride_fh,
    stride_fb,
    stride_fl,
    stride_tb,
    stride_tn,
    stride_ttb,
    stride_ttn,
    batch_size: tl.int32,
    max_num_blocks_per_seq: tl.int32,
    block_size: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_idx = pid % batch_size
    head_idx = pid // batch_size

    score_base = score_cache_ptr + head_idx * stride_sh
    flag_base = flag_ptr + batch_idx * stride_fz + head_idx * stride_fh
    block_id = tl.load(target_block_table_ptr + batch_idx * stride_ttb)
    score_save_ptr = score_base + block_id * stride_sb
    pos = 0
    for block_idx in range(max_num_blocks_per_seq):
        block_id = tl.load(
            block_table_ptr + block_idx * stride_tn + batch_idx * stride_tb
        )
        if not block_id == -1:
            if block_id < 0:
                block_id = -block_id - 2
            for slot_idx in range(block_size):
                flag = tl.load(flag_base + block_idx * stride_fb + slot_idx * stride_fl)
                if flag:
                    s = tl.load(
                        score_base + block_id * stride_sb + slot_idx * stride_sl
                    )
                    tl.store(score_save_ptr, s)
                    pos += 1
                    if pos % block_size == 0:
                        next_block_id = tl.load(
                            target_block_table_ptr
                            + (pos // block_size) * stride_ttn
                            + batch_idx * stride_ttb
                        )
                        score_save_ptr = score_base + next_block_id * stride_sb
                    else:
                        score_save_ptr = score_save_ptr + stride_sl


def compress_score(
    score_cache: torch.Tensor,
    flag: torch.Tensor,
    block_table: torch.Tensor,
    target_block_table: torch.Tensor,
):
    """
    compress score in score_cache

    Arguments:
        score_cache: (num_kvcache_blocks, block_size, num_kv_heads)
        flag: (batch_size, num_kv_heads, max_num_blocks_per_seq, block_size)
        block_table: (batch_size, max_num_blocks_per_seq)
        target_block_table: (batch_size, max_num_blocks_per_seq-2)
    """
    batch_size, num_kv_heads, max_num_blocks_per_seq, block_size = flag.shape
    grid = (batch_size * num_kv_heads,)
    compress_score_kernel[grid](
        score_cache,
        flag,
        block_table,
        target_block_table,
        **_strides(score_cache, "sb", "sl", "sh"),
        **_strides(flag, "fz", "fh", "fb", "fl"),
        **_strides(block_table, "tb", "tn"),
        **_strides(target_block_table, "ttb", "ttn"),
        batch_size=batch_size,
        max_num_blocks_per_seq=max_num_blocks_per_seq,
        block_size=block_size,
    )
    return score_cache
