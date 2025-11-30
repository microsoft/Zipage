import triton
import triton.language as tl
import torch

from zipvllm.kernel.utils import _strides
from zipvllm.kernel.key_norm import key_norm


@triton.jit
def light_similarity_score_kernel(
    key_cache_ptr,
    similarity_cos_ptr,
    norm_ptr,
    last_block_norm_ptr,
    block_table_ptr,
    last_block_ptr,
    stride_kb,
    stride_kl,
    stride_kh,
    stride_kd,
    stride_sz,
    stride_sh,
    stride_sb,
    stride_sl,
    stride_nz,
    stride_nh,
    stride_nb,
    stride_nl,
    stride_lnz,
    stride_lnh,
    stride_lnb,
    stride_lnl,
    stride_tz,
    stride_tb,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    max_num_blocks_per_seq: tl.int32,
    block_size: tl.constexpr,
    batch_size: tl.int32,
    head_dim: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_idx = pid % batch_size
    rem = pid // batch_size
    head_idx = rem // max_num_blocks_per_seq
    m_block_idx = rem % max_num_blocks_per_seq
    m_block_id = tl.load(
        block_table_ptr + batch_idx * stride_tz + m_block_idx * stride_tb
    )
    if not m_block_id == -1:
        if m_block_id < 0:
            m_block_id = -m_block_id - 2
        for m_block_offset in range(0, block_size, BLOCK_M):
            m_key_ptr = tl.make_block_ptr(
                base=key_cache_ptr + m_block_id * stride_kb + head_idx * stride_kh,
                shape=(block_size, head_dim),
                strides=(stride_kl, stride_kd),
                offsets=(m_block_offset, 0),
                block_shape=(BLOCK_M, head_dim),
                order=(1, 0),
            )
            m_key = tl.load(m_key_ptr, boundary_check=(0,))
            raw_dtype = m_key.dtype
            m_norm_ptr = tl.make_block_ptr(
                base=norm_ptr
                + batch_idx * stride_nz
                + head_idx * stride_nh
                + m_block_idx * stride_nb,
                shape=(block_size, 1),
                strides=(stride_nl, 1),
                offsets=(m_block_offset, 0),
                block_shape=(BLOCK_M, 1),
                order=(1, 0),
            )
            m_norm = tl.load(m_norm_ptr, boundary_check=(0,))
            m_key = m_key / (m_norm)
            s = tl.zeros((BLOCK_M, 1), dtype=tl.float32)
            last_block_id = tl.load(last_block_ptr + batch_idx)

            for n_block_offset in range(0, block_size, BLOCK_N):
                n_key_ptr = tl.make_block_ptr(
                    base=key_cache_ptr
                    + last_block_id * stride_kb
                    + head_idx * stride_kh,
                    shape=(head_dim, block_size),
                    strides=(stride_kd, stride_kl),
                    offsets=(0, n_block_offset),
                    block_shape=(head_dim, BLOCK_N),
                    order=(0, 1),
                )
                n_key = tl.load(n_key_ptr, boundary_check=(1,))
                n_norm_ptr = tl.make_block_ptr(
                    base=last_block_norm_ptr
                    + batch_idx * stride_lnz
                    + head_idx * stride_lnh,
                    shape=(1, block_size),
                    strides=(1, stride_nl),
                    offsets=(0, n_block_offset),
                    block_shape=(1, BLOCK_N),
                    order=(0, 1),
                )
                n_norm = tl.load(n_norm_ptr, boundary_check=(0,))
                n_key = n_key / (n_norm)
                similarity = tl.dot(m_key, n_key)
                similarity = tl.sum(similarity, axis=1, keep_dims=True)
                s += similarity
            s_ptr = tl.make_block_ptr(
                base=similarity_cos_ptr
                + batch_idx * stride_sz
                + head_idx * stride_sh
                + m_block_idx * stride_sb,
                shape=(block_size, 1),
                strides=(stride_sl, 1),
                offsets=(m_block_offset, 0),
                block_shape=(BLOCK_M, 1),
                order=(1, 0),
            )
            tl.store(s_ptr, s.to(raw_dtype), boundary_check=(0,))


def light_similarity_score(
    key_cache: torch.Tensor,
    block_table: torch.Tensor,
    last_block: torch.Tensor,
    temperature: float = 1.0,
    debug: bool = False,
):
    """
    Calculate cos similarity score between key cache.
    Args:
        key_cache: (num_kvcache_blocks, block_size, num_kv_heads, head_dim)
        block_table: (batch_size, max_num_blocks_per_seq)
        last_block: (batch_size)
        retain_ratio: float
        threshold: float
    Returns:
        similarity_score: (batch_size, num_kv_heads, max_num_blocks_per_seq, block_size)
    """
    BLOCK_M = 16
    BLOCK_N = 64

    _, block_size, num_kv_heads, head_dim = key_cache.shape
    batch_size, max_num_blocks_per_seq = block_table.shape

    norm = key_norm(key_cache, block_table)
    last_block_norm = key_norm(key_cache, last_block.unsqueeze(1))

    similarity_cos = torch.full(
        (batch_size, num_kv_heads, max_num_blocks_per_seq, block_size),
        -float("inf"),
        device=key_cache.device,
        dtype=key_cache.dtype,
    )
    grid = (batch_size * num_kv_heads * max_num_blocks_per_seq,)

    light_similarity_score_kernel[grid](
        key_cache,
        similarity_cos,
        norm,
        last_block_norm,
        block_table,
        last_block,
        **_strides(key_cache, "kb", "kl", "kh", "kd"),
        **_strides(similarity_cos, "sz", "sh", "sb", "sl"),
        **_strides(norm, "nz", "nh", "nb", "nl"),
        **_strides(last_block_norm, "lnz", "lnh", "lnb", "lnl"),
        **_strides(block_table, "tz", "tb"),
        max_num_blocks_per_seq=max_num_blocks_per_seq,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        block_size=block_size,
        batch_size=batch_size,
        head_dim=head_dim,
    )
    del norm
    del last_block_norm

    if debug:
        logits = similarity_cos.clone()

    similarity_cos = similarity_cos.view(batch_size, num_kv_heads, -1)
    seq_length = (block_table != -1).sum(dim=-1) * block_size
    similarity_cos = similarity_cos.div_(
        temperature * seq_length.unsqueeze(-1).unsqueeze(-1)
    )
    similarity_cos = similarity_cos - similarity_cos.max(dim=-1, keepdim=True).values
    similarity_cos = similarity_cos.softmax(dim=-1)
    similarity_cos = similarity_cos.reshape(
        batch_size, num_kv_heads, max_num_blocks_per_seq, block_size
    )
    if debug:
        return similarity_cos, logits
    return similarity_cos
