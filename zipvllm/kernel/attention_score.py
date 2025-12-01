import triton
import triton.language as tl
import torch
from zipvllm.kernel.utils import get_padded_headsize, _strides


@triton.jit
def attention_score_kernel(
    k_cache_ptr,
    query_cache_ptr,
    qk_buffer_ptr,
    seq_idx_ptr,
    block_table_ptr,
    stride_qy: tl.int64,
    stride_qs,
    stride_qc,
    stride_qh,
    stride_qd,
    stride_ky: tl.int64,
    stride_kb,
    stride_kc,
    stride_kh,
    stride_kd,
    stride_sy: tl.int64,
    stride_sb,
    stride_sh,
    stride_sc,
    stride_sm,
    stride_sd,
    stride_tb,
    stride_tn,
    H_q: tl.constexpr,
    H_kv: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    ACTUAL_BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_GQA: tl.constexpr,
    softmax_scale: tl.float32,
    BLOCK_SIZE: tl.constexpr,
    QUERY_CACHE_LEN: tl.constexpr,
    max_num_blocks_per_seq: tl.int32,
):
    pid = tl.program_id(0)
    layer_id = tl.program_id(1)
    num_q_blocks = tl.cdiv(QUERY_CACHE_LEN, BLOCK_M)
    bsz = pid // (H_q * num_q_blocks)
    rem = pid % (H_q * num_q_blocks)
    head_id = rem // num_q_blocks
    q_id = rem % num_q_blocks

    seq_id = tl.load(seq_idx_ptr + bsz)

    HEAD_RATIO: tl.constexpr = H_q // H_kv
    if IS_GQA:
        k_head_idx = head_id // HEAD_RATIO
    else:
        k_head_idx = head_id

    Q_base = (
        query_cache_ptr
        + seq_id * stride_qs
        + head_id * stride_qh
        + layer_id * stride_qy
    )

    Q_block_ptr = tl.make_block_ptr(
        base=Q_base,
        shape=(QUERY_CACHE_LEN, ACTUAL_BLOCK_DMODEL),
        strides=(stride_qc, stride_qd),
        offsets=(q_id * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    q = tl.load(Q_block_ptr, boundary_check=(0, 1))

    for block_idx in range(max_num_blocks_per_seq):
        block_id = tl.load(block_table_ptr + bsz * stride_tb + block_idx * stride_tn)
        if not block_id == -1:
            if block_id < 0:
                block_id = -block_id - 2
                last_block = True
            else:
                last_block = False
            qk_base = (
                qk_buffer_ptr
                + bsz * stride_sb
                + head_id * stride_sh
                + block_idx * stride_sm
                + layer_id * stride_sy
            )
            k_base = (
                k_cache_ptr
                + block_id * stride_kb
                + k_head_idx * stride_kh
                + layer_id * stride_ky
            )
            for K_OFFSET in range(0, BLOCK_SIZE, BLOCK_N):
                K_block_ptr = tl.make_block_ptr(
                    base=k_base,
                    shape=(ACTUAL_BLOCK_DMODEL, BLOCK_SIZE),
                    strides=(stride_kd, stride_kc),
                    offsets=(0, K_OFFSET),
                    block_shape=(BLOCK_DMODEL, BLOCK_N),
                    order=(0, 1),
                )
                k = tl.load(K_block_ptr, boundary_check=(0, 1))
                qk = tl.dot(q, k)
                qk = qk * softmax_scale
                if last_block:
                    col_index = (K_OFFSET + tl.arange(0, BLOCK_N))[None, :]
                    raw_index = tl.arange(
                        BLOCK_SIZE - QUERY_CACHE_LEN,
                        BLOCK_SIZE - QUERY_CACHE_LEN + BLOCK_M,
                    )[:, None]
                    raw_index = raw_index + q_id * BLOCK_M
                    mask = col_index > raw_index
                    qk = tl.where(mask, -float("inf"), qk)
                qk_block_ptr = tl.make_block_ptr(
                    base=qk_base,
                    shape=(QUERY_CACHE_LEN, BLOCK_SIZE),
                    strides=(stride_sc, stride_sd),
                    offsets=(q_id * BLOCK_M, K_OFFSET),
                    block_shape=(BLOCK_M, BLOCK_N),
                    order=(1, 0),
                )
                qk = qk.to(q.dtype)
                tl.store(qk_block_ptr, qk, boundary_check=(0, 1))


def attention_score(
    k_cache: torch.Tensor,
    query_cache: torch.Tensor,
    seq_idx: torch.Tensor,
    block_table: torch.Tensor,
    softmax_scale=None,
    reduced="mean",
    return_logits=False,
) -> torch.Tensor:
    """
    calculate attention scores, each sequence each layer's attention score is num_attention_heads * query_cache_len * cache_seqlens
    reduce the attention scores to num_kv_heads * query_cache_len * cache_seqlens by max reduce along query head
    then reduce the attention scores to num_attention_heads * cache_seqlens by mean reduce
    Arguments:
        k_cache: Shape (num_layers, num_kvcache_blocks, block_size, num_kv_heads, head_dim) cached key
        query_cache: Shape (num_layers, max_num_seqs, query_cache_len, num_attention_heads, head_dim) cached query
        seq_idx: Shape (batch_size,) query cache index for each sequence
        block_table: Shape (batch_size, max_num_blocks_per_seq) each sequence KV cache actual position, -1 is padding
        softmax_scale: float
    Returns:
        reduced_scores: Shape (num_layers, batch_size, num_kv_heads, max_num_blocks_per_seq, block_size)
    """
    # kernel config
    BLOCK_M = 16
    BLOCK_N = 64

    # get dims
    num_layers, num_kvcache_blocks, block_size, num_kv_heads, head_dim = k_cache.shape
    _, _, query_cache_len, num_attention_heads, _ = query_cache.shape
    batch_size, max_num_blocks_per_seq = block_table.shape
    BLOCK_HEAD_DIM = get_padded_headsize(head_dim)
    if softmax_scale is None:
        softmax_scale = query_cache.shape[-1] ** (-0.5)
    G_q = num_attention_heads // num_kv_heads
    IS_GQA = G_q > 1

    assert k_cache.is_contiguous()
    assert query_cache.is_contiguous()
    assert seq_idx.is_contiguous()
    assert block_table.is_contiguous()

    qk_buffer = torch.full(
        (
            num_layers,
            batch_size,
            num_attention_heads,
            query_cache_len,
            max_num_blocks_per_seq,
            block_size,
        ),
        -float("inf"),
        dtype=k_cache.dtype,
        device=k_cache.device,
    )

    grid_qk = (
        triton.cdiv(query_cache_len, BLOCK_M) * batch_size * num_attention_heads,
        num_layers,
    )

    attention_score_kernel[grid_qk](
        k_cache_ptr=k_cache,
        query_cache_ptr=query_cache,
        qk_buffer_ptr=qk_buffer,
        seq_idx_ptr=seq_idx,
        block_table_ptr=block_table,
        **_strides(query_cache, "qy", "qs", "qc", "qh", "qd"),
        **_strides(k_cache, "ky", "kb", "kc", "kh", "kd"),
        **_strides(qk_buffer, "sy", "sb", "sh", "sc", "sm", "sd"),
        **_strides(block_table, "tb", "tn"),
        H_q=num_attention_heads,
        H_kv=num_kv_heads,
        BLOCK_M=BLOCK_M,
        BLOCK_DMODEL=BLOCK_HEAD_DIM,
        ACTUAL_BLOCK_DMODEL=head_dim,
        BLOCK_N=BLOCK_N,
        IS_GQA=IS_GQA,
        softmax_scale=softmax_scale,
        BLOCK_SIZE=block_size,
        QUERY_CACHE_LEN=query_cache_len,
        max_num_blocks_per_seq=max_num_blocks_per_seq,
    )

    qk_buffer = qk_buffer.reshape(
        num_layers, batch_size, num_attention_heads, query_cache_len, -1
    )
    if return_logits:
        logits = qk_buffer.clone()
    dtype = qk_buffer.dtype
    qk_buffer = qk_buffer.float()
    score = torch.softmax(
        qk_buffer - qk_buffer.max(dim=-1, keepdim=True).values, dim=-1
    )
    score = score.to(dtype)
    del qk_buffer
    if IS_GQA:
        score = score.view(
            num_layers, batch_size, num_kv_heads, G_q, query_cache_len, -1
        )
        reduced_scores = torch.max(score, dim=3).values
    else:
        reduced_scores = score
    del score
    reduced_scores = reduced_scores.reshape(
        num_layers,
        batch_size,
        num_kv_heads,
        query_cache_len,
        max_num_blocks_per_seq,
        block_size,
    )

    if reduced == "mean":
        reduced_scores = reduced_scores.mean(dim=3)
    elif reduced == "max":
        reduced_scores = reduced_scores.max(dim=3).values
    else:
        raise ValueError(f"Invalid reduction method: {reduced}")
    if return_logits:
        return reduced_scores, logits
    return reduced_scores
