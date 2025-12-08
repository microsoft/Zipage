import triton
import triton.language as tl
import torch


@triton.jit
def store_query_cache_kernel(
    query_ptr,
    query_cache_ptr,
    slot_mapping_ptr,
    stride_ql,
    stride_cs,
    stride_cl,
    stride_sn,
    stride_sm,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    idx = tl.program_id(0)
    pos = tl.load(slot_mapping_ptr + idx * stride_sn)
    seq_id = tl.load(slot_mapping_ptr + idx * stride_sn + stride_sm)
    slot = tl.load(slot_mapping_ptr + idx * stride_sn + stride_sm * 2)
    offs = tl.arange(0, BLOCK_D)
    for offset in range(0, D, BLOCK_D):
        query_offsets = offs + offset
        mask = offs + offset < D
        q = tl.load(query_ptr + pos * stride_ql + query_offsets, mask=mask)
        tl.store(
            query_cache_ptr + seq_id * stride_cs + slot * stride_cl + query_offsets,
            q,
            mask=mask,
        )


def store_query_cache(
    query: torch.Tensor, query_cache: torch.Tensor, slot_mapping: torch.Tensor
):
    """
    Store query cache into query cache tensor.
    Args:
        query: (L, num_heads, head_dim)
        query_cache: (max_num_seqs, query_cache_len, num_heads, head_dim)
        slot_mapping: (N,3)
    """
    BLOCK_D = 128 * 8
    N = slot_mapping.shape[0]
    _, num_heads, head_dim = query.shape
    D = num_heads * head_dim

    if query_cache.numel() == 0: # warmup
        return
        
    assert query.stride(-1) == 1
    assert query.stride(1) == head_dim
    assert query_cache.stride(1) == D
    store_query_cache_kernel[(N,)](
        query,
        query_cache,
        slot_mapping,
        query.stride(0),
        query_cache.stride(0),
        query_cache.stride(1),
        slot_mapping.stride(0),
        slot_mapping.stride(1),
        D=D,
        BLOCK_D=BLOCK_D,
        num_warps=1,
    )
