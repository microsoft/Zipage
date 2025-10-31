import triton
import triton.language as tl
import torch
from zipvllm.kernel.utils import _strides


@triton.jit
def compress_kv_out_order_kernel(
    key_ptr,
    value_ptr,
    save_indices_ptr,
    load_indices_ptr,
    block_table_ptr,
    stride_kb,
    stride_ks,
    stride_kh,
    stride_kd,
    stride_vb,
    stride_vs,
    stride_vh,
    stride_vd,
    stride_tb,
    stride_tn,
    BLOCK_D: tl.constexpr,
    MODEL_D: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_idx = tl.load(save_indices_ptr + pid * 4)
    head_idx = tl.load(save_indices_ptr + pid * 4 + 1)

    save_block_idx = tl.load(save_indices_ptr + pid * 4 + 2)
    save_block_id = tl.load(
        block_table_ptr + batch_idx * stride_tb + save_block_idx * stride_tn
    )
    save_slot_idx = tl.load(save_indices_ptr + pid * 4 + 3)

    save_key_base = (
        key_ptr
        + head_idx * stride_kh
        + save_block_id * stride_kb
        + save_slot_idx * stride_ks
    )
    save_value_base = (
        value_ptr
        + head_idx * stride_vh
        + save_block_id * stride_vb
        + save_slot_idx * stride_vs
    )

    load_block_idx = tl.load(load_indices_ptr + pid * 4 + 2)
    load_block_id = tl.load(
        block_table_ptr + batch_idx * stride_tb + load_block_idx * stride_tn
    )
    if load_block_id < 0:
        load_block_id = -load_block_id - 2
    load_slot_idx = tl.load(load_indices_ptr + pid * 4 + 3)

    load_key_base = (
        key_ptr
        + head_idx * stride_kh
        + load_block_id * stride_kb
        + load_slot_idx * stride_ks
    )
    load_value_base = (
        value_ptr
        + head_idx * stride_vh
        + load_block_id * stride_vb
        + load_slot_idx * stride_vs
    )

    offs = tl.arange(0, BLOCK_D)

    for d_offset in range(0, MODEL_D, BLOCK_D):
        d_offs = offs + d_offset
        k = tl.load(load_key_base + d_offs * stride_kd, mask=d_offs < MODEL_D)
        v = tl.load(load_value_base + d_offs * stride_vd, mask=d_offs < MODEL_D)
        tl.store(save_key_base + d_offs * stride_kd, k, mask=d_offs < MODEL_D)
        tl.store(save_value_base + d_offs * stride_vd, v, mask=d_offs < MODEL_D)


def compress_kv_out_order(
    key: torch.Tensor,
    value: torch.Tensor,
    save_indices: torch.Tensor,
    load_indices: torch.Tensor,
    block_table: torch.Tensor,
):
    """
    compress key and value

    Arguments:
        key: (num_kvcache_blocks, block_size, num_kv_heads, head_dim)
        value: (num_kvcache_blocks, block_size, num_kv_heads, head_dim)
        save_indices: (N, 4)
        load_indices: (N, 4)
        block_table: (batch_size, max_num_blocks_per_seq)
    """
    BLOCK_D = 128
    N = save_indices.shape[0]
    _, block_size, num_kv_heads, head_dim = key.shape
    grid = (N,)
    compress_kv_out_order_kernel[grid](
        key,
        value,
        save_indices,
        load_indices,
        block_table,
        **_strides(key, "kb", "ks", "kh", "kd"),
        **_strides(value, "vb", "vs", "vh", "vd"),
        **_strides(block_table, "tb", "tn"),
        BLOCK_D=BLOCK_D,
        MODEL_D=head_dim,
    )
