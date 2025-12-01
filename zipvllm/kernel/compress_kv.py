import triton
import triton.language as tl
import torch
from zipvllm.kernel.utils import _strides


@triton.jit
def compress_kv_kernel(
    key_ptr,
    value_ptr,
    flag_ptr,
    block_table_ptr,
    target_block_table_ptr,
    stride_ky:tl.int64,
    stride_kb,
    stride_ks,
    stride_kh,
    stride_kd,
    stride_vy:tl.int64,
    stride_vb,
    stride_vs,
    stride_vh,
    stride_vd,
    stride_fy:tl.int64,
    stride_fz,
    stride_fh,
    stride_fb,
    stride_fs,
    stride_tb,
    stride_tn,
    stride_ttb,
    stride_ttn,
    BLOCK_D: tl.constexpr,
    MODEL_D: tl.constexpr,
    block_size: tl.constexpr,
    batch_size: tl.int32,
    max_num_blocks_per_seq: tl.int32,
):
    pid = tl.program_id(0)
    layer_id = tl.program_id(1)
    batch_idx = pid % batch_size
    head_idx = pid // batch_size

    key_base = key_ptr + head_idx * stride_kh + layer_id * stride_ky
    value_base = value_ptr + head_idx * stride_vh + layer_id * stride_vy
    flag_base = (
        flag_ptr + batch_idx * stride_fz + head_idx * stride_fh + layer_id * stride_fy
    )

    block_id = tl.load(target_block_table_ptr + batch_idx * stride_ttb)
    key_save_ptr = key_base + block_id * stride_kb
    value_save_ptr = value_base + block_id * stride_vb
    pos = 0
    offs = tl.arange(0, BLOCK_D)
    for block_idx in range(max_num_blocks_per_seq):
        block_id = tl.load(
            block_table_ptr + block_idx * stride_tn + batch_idx * stride_tb
        )
        if not block_id == -1:
            if block_id < 0:
                block_id = -block_id - 2
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
                        tl.store(
                            key_save_ptr + d_offs * stride_kd, k, mask=d_offs < MODEL_D
                        )
                        tl.store(
                            value_save_ptr + d_offs * stride_vd,
                            v,
                            mask=d_offs < MODEL_D,
                        )
                    pos += 1
                    if pos % block_size == 0:
                        next_block_id = tl.load(
                            target_block_table_ptr
                            + (pos // block_size) * stride_ttn
                            + batch_idx * stride_ttb
                        )
                        key_save_ptr = key_base + next_block_id * stride_kb
                        value_save_ptr = value_base + next_block_id * stride_vb
                    else:
                        key_save_ptr = key_save_ptr + stride_ks
                        value_save_ptr = value_save_ptr + stride_vs


def compress_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    flag: torch.Tensor,
    block_table: torch.Tensor,
    target_block_table: torch.Tensor,
):
    """
    compress key and value to k_cache and v_cache

    Arguments:
        key: (num_layers, num_kvcache_blocks, block_size, num_kv_heads, head_dim)
        value: (num_layers, num_kvcache_blocks, block_size, num_kv_heads, head_dim)
        flag: (num_layers, batch_size, num_kv_heads, max_num_blocks_per_seq, block_size)
        block_table: (batch_size, max_num_blocks_per_seq)
        target_block_table: (batch_size, max_num_blocks_per_seq-2)
    """
    BLOCK_D = 128
    num_layers, _, block_size, num_kv_heads, head_dim = key.shape
    _, batch_size, _, max_num_blocks_per_seq, _ = flag.shape
    grid = (batch_size * num_kv_heads, num_layers)
    compress_kv_kernel[grid](
        key,
        value,
        flag,
        block_table,
        target_block_table,
        **_strides(key, "ky", "kb", "ks", "kh", "kd"),
        **_strides(value, "vy", "vb", "vs", "vh", "vd"),
        **_strides(flag, "fy", "fz", "fh", "fb", "fs"),
        **_strides(block_table, "tb", "tn"),
        **_strides(target_block_table, "ttb", "ttn"),
        BLOCK_D=BLOCK_D,
        MODEL_D=head_dim,
        block_size=block_size,
        batch_size=batch_size,
        max_num_blocks_per_seq=max_num_blocks_per_seq,
        num_warps=1,
    )
