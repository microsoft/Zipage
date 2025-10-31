import triton
import triton.language as tl
import torch
from compressvllm.kernel.utils import _strides


@triton.jit
def compress_kv_kernel(
    key_ptr,
    value_ptr,
    flag_ptr,
    block_table_ptr,
    stride_kb,
    stride_ks,
    stride_kh,
    stride_kd,
    stride_vb,
    stride_vs,
    stride_vh,
    stride_vd,
    stride_fz,
    stride_fh,
    stride_fb,
    stride_fs,
    stride_tb,
    stride_tn,
    BLOCK_D: tl.constexpr,
    MODEL_D: tl.constexpr,
    batch_size: tl.constexpr,
    block_size: tl.constexpr,
    max_num_blocks_per_seq: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_idx = pid // batch_size
    head_idx = pid % batch_size

    key_base = key_ptr + head_idx * stride_kh
    value_base = value_ptr + head_idx * stride_vh
    flag_base = flag_ptr + batch_idx * stride_fz + head_idx * stride_fh

    block_id = tl.load(block_table_ptr + batch_idx * stride_tb)
    key_save_ptr = key_base + block_id * stride_kb
    value_save_ptr = value_base + block_id * stride_vb
    pos = 0
    offs = tl.arange(0, BLOCK_D)
    for block_idx in range(max_num_blocks_per_seq):
        block_id = tl.load(
            block_table_ptr + block_idx * stride_tn + batch_idx * stride_tb
        )
        if block_id >= 0:
            for slot_idx in range(block_size):
                flag = tl.load(flag_base + block_idx * stride_fb + slot_idx * stride_fs)
                if flag:
                    for d_offset in range(0, MODEL_D, BLOCK_D):
                        d_offs = offs + d_offset
                        k = tl.load(
                            key_base
                            + block_id * stride_kb
                            + slot_idx * stride_ks
                            + d_offs * stride_kd,
                            mask=d_offs < MODEL_D,
                        )
                        v = tl.load(
                            value_base
                            + block_id * stride_vb
                            + slot_idx * stride_vs
                            + d_offs * stride_vd,
                            mask=d_offs < MODEL_D,
                        )
                        tl.store(key_save_ptr + d_offs * stride_kd, k)
                        tl.store(value_save_ptr + d_offs * stride_vd, v)
                    pos += 1
                    if pos % block_size == 0:
                        next_block_id = tl.load(
                            block_table_ptr
                            + (pos // block_size) * stride_tn
                            + batch_idx * stride_tb
                        )
                        key_save_ptr = key_base + next_block_id * stride_kb
                        value_save_ptr = value_base + next_block_id * stride_vb
                    else:
                        key_save_ptr = key_save_ptr + stride_ks
                        value_save_ptr = value_save_ptr + stride_vs


@torch.compile
def topk_mask(input: torch.Tensor, k: int, dim: int = -1):
    """
    calculate if the last dimension of input is top-k, return a bool mask with the same shape
    top-k is True, others are False
    """
    topk_indices = torch.topk(input, k, dim=dim)[1]
    mask = torch.zeros_like(input, dtype=torch.bool)
    mask.scatter_(dim, topk_indices, True)
    return mask


@triton.jit
def compress_score_kernel(
    score_cache_ptr,
    flag_ptr,
    block_table_ptr,
    stride_sb,
    stride_sl,
    stride_sh,
    stride_fz,
    stride_fh,
    stride_fb,
    stride_fl,
    stride_tb,
    stride_tn,
    batch_size: tl.constexpr,
    max_num_blocks_per_seq: tl.constexpr,
    block_size: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_idx = pid // batch_size
    head_idx = pid % batch_size

    score_base = score_cache_ptr + head_idx * stride_sh
    flag_base = flag_ptr + batch_idx * stride_fz + head_idx * stride_fh
    block_id = tl.load(block_table_ptr + batch_idx * stride_tb)
    score_save_ptr = score_base + block_id * stride_sb
    pos = 0
    for block_idx in range(max_num_blocks_per_seq):
        block_id = tl.load(
            block_table_ptr + block_idx * stride_tn + batch_idx * stride_tb
        )
        if block_id >= 0:
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
                            block_table_ptr
                            + (pos // block_size) * stride_tn
                            + batch_idx * stride_tb
                        )
                        score_save_ptr = score_base + next_block_id * stride_sb
                    else:
                        score_save_ptr = score_save_ptr + stride_sl
        elif not block_id == -1:
            block_id = -block_id - 1
            offs = tl.arange(0, block_size)
            tl.store(score_base + offs * stride_sl + block_id * stride_sb, 0)


def compress_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    flag: torch.Tensor,
    block_table: torch.Tensor,
):
    """
    compress key and value to k_cache and v_cache

    Arguments:
        key: (num_kvcache_blocks, block_size, num_kv_heads, head_dim)
        value: (num_kvcache_blocks, block_size, num_kv_heads, head_dim)
        flag: (batch_size, num_kv_heads, max_num_blocks_per_seq, block_size)
        block_table: (batch_size, max_num_blocks_per_seq)
    """
    BLOCK_D = 128
    _, block_size, num_kv_heads, head_dim = key.shape
    batch_size, _, max_num_blocks_per_seq, _ = flag.shape
    grid = (batch_size * num_kv_heads,)
    compress_kv_kernel[grid](
        key,
        value,
        flag,
        block_table,
        **_strides(key, "kb", "ks", "kh", "kd"),
        **_strides(value, "vb", "vs", "vh", "vd"),
        **_strides(flag, "fz", "fh", "fb", "fs"),
        **_strides(block_table, "tb", "tn"),
        BLOCK_D=BLOCK_D,
        MODEL_D=head_dim,
        batch_size=batch_size,
        block_size=block_size,
        max_num_blocks_per_seq=max_num_blocks_per_seq,
    )


def compress_score(
    score_cache: torch.Tensor, flag: torch.Tensor, block_table: torch.Tensor
):
    """
    compress score in score_cache and set the free block to zeros

    Arguments:
        score_cache: (num_kvcache_blocks, block_size, num_kv_heads)
        flag: (batch_size, num_kv_heads, max_num_blocks_per_seq, block_size)
        block_table: (batch_size, max_num_blocks_per_seq)
    """
    batch_size, num_kv_heads, max_num_blocks_per_seq, block_size = flag.shape
    grid = (batch_size * num_kv_heads,)
    compress_score_kernel[grid](
        score_cache,
        flag,
        block_table,
        **_strides(score_cache, "sb", "sl", "sh"),
        **_strides(flag, "fz", "fh", "fb", "fl"),
        **_strides(block_table, "tb", "tn"),
        batch_size=batch_size,
        max_num_blocks_per_seq=max_num_blocks_per_seq,
        block_size=block_size,
    )
    return score_cache
