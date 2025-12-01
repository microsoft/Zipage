import triton
import triton.language as tl
import torch

from zipvllm.kernel.utils import _strides
from zipvllm.kernel.key_norm import key_norm


@triton.jit
def raw_similarity_score_kernel(
    key_cache_ptr,
    similarity_cos_ptr,
    key_norm_ptr,
    zero_out_ptr,
    block_table_ptr,
    stride_ky:tl.int64,
    stride_kb,
    stride_kl,
    stride_kh,
    stride_kd,
    stride_sy:tl.int64,
    stride_sz,
    stride_sh,
    stride_sb,
    stride_sl,
    stride_ny:tl.int64,
    stride_nz,
    stride_nh,
    stride_nb,
    stride_nl,
    stride_zy:tl.int64,
    stride_zz,
    stride_zh,
    stride_zb,
    stride_zl,
    stride_tz,
    stride_tb,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    max_num_blocks_per_seq: tl.int32,
    block_size: tl.constexpr,
    batch_size: tl.int32,
    head_dim: tl.constexpr,
    threshold: tl.float32,
):
    pid = tl.program_id(0)
    layer_id = tl.program_id(1)
    batch_idx = pid % batch_size
    head_idx = pid // batch_size

    for m_block_idx in range(max_num_blocks_per_seq - 1, -1, -1):
        m_block_id = tl.load(
            block_table_ptr + batch_idx * stride_tz + m_block_idx * stride_tb
        )
        if not m_block_id == -1:
            if m_block_id < 0:
                m_block_id = -m_block_id - 2
            for m_block_offset in range(block_size - BLOCK_M, -1, -BLOCK_M):
                m_key_ptr = tl.make_block_ptr(
                    base=key_cache_ptr
                    + m_block_id * stride_kb
                    + head_idx * stride_kh
                    + layer_id * stride_ky,
                    shape=(block_size, head_dim),
                    strides=(stride_kl, stride_kd),
                    offsets=(m_block_offset, 0),
                    block_shape=(BLOCK_M, head_dim),
                    order=(1, 0),
                )
                m_key = tl.load(m_key_ptr, boundary_check=(0,))
                raw_dtype = m_key.dtype
                m_norm_ptr = tl.make_block_ptr(
                    base=key_norm_ptr
                    + batch_idx * stride_nz
                    + head_idx * stride_nh
                    + m_block_idx * stride_nb
                    + layer_id * stride_ny,
                    shape=(block_size, 1),
                    strides=(stride_nl, 1),
                    offsets=(m_block_offset, 0),
                    block_shape=(BLOCK_M, 1),
                    order=(1, 0),
                )
                m_norm = tl.load(m_norm_ptr, boundary_check=(0,))
                m_key = m_key / (m_norm)
                s = tl.zeros((BLOCK_M, 1), dtype=tl.float32)
                m_indices = (
                    m_block_idx * block_size
                    + m_block_offset
                    + tl.arange(0, BLOCK_M)[:, None]
                )

                for n_block_idx in range(max_num_blocks_per_seq):
                    n_block_id = tl.load(
                        block_table_ptr
                        + batch_idx * stride_tz
                        + n_block_idx * stride_tb
                    )
                    if not n_block_id == -1:
                        if n_block_id < 0:
                            n_block_id = -n_block_id - 2
                        for n_block_offset in range(0, block_size, BLOCK_N):
                            n_key_ptr = tl.make_block_ptr(
                                base=key_cache_ptr
                                + n_block_id * stride_kb
                                + head_idx * stride_kh
                                + layer_id * stride_ky,
                                shape=(head_dim, block_size),
                                strides=(stride_kd, stride_kl),
                                offsets=(0, n_block_offset),
                                block_shape=(head_dim, BLOCK_N),
                                order=(0, 1),
                            )
                            n_key = tl.load(n_key_ptr, boundary_check=(1,))
                            n_norm_ptr = tl.make_block_ptr(
                                base=key_norm_ptr
                                + batch_idx * stride_nz
                                + head_idx * stride_nh
                                + n_block_idx * stride_nb
                                + layer_id * stride_ny,
                                shape=(1, block_size),
                                strides=(1, stride_nl),
                                offsets=(0, n_block_offset),
                                block_shape=(1, BLOCK_N),
                                order=(0, 1),
                            )
                            n_norm = tl.load(n_norm_ptr, boundary_check=(0,))
                            n_key = n_key / (n_norm)
                            similarity = tl.dot(m_key, n_key)
                            n_indices = (
                                n_block_idx * block_size
                                + n_block_offset
                                + tl.arange(0, BLOCK_N)[None, :]
                            )
                            same_key_mask = m_indices == n_indices
                            similarity = tl.where(same_key_mask, 0.0, similarity)

                            zo_ptr = tl.make_block_ptr(
                                base=zero_out_ptr
                                + batch_idx * stride_zz
                                + head_idx * stride_zh
                                + n_block_idx * stride_zb
                                + layer_id * stride_zy,
                                shape=(1, block_size),
                                strides=(1, stride_zl),
                                offsets=(0, n_block_offset),
                                block_shape=(1, BLOCK_N),
                                order=(0, 1),
                            )
                            zo = tl.load(zo_ptr, boundary_check=(1,))
                            threshold_mask = ((similarity > threshold) & (~zo)).to(
                                tl.int1
                            )
                            row_indices = tl.arange(0, BLOCK_M)[:, None]
                            max_idx_per_col = tl.max(
                                tl.where(
                                    threshold_mask,
                                    row_indices,
                                    -1,
                                ),
                                axis=0,
                                keep_dims=True,
                            )
                            last_threshold_mask = row_indices == max_idx_per_col
                            similarity = tl.where(last_threshold_mask, 0.0, similarity)
                            last_threshold_mask = tl.max(
                                last_threshold_mask, axis=0, keep_dims=True
                            )
                            last_threshold_mask = last_threshold_mask | zo
                            last_threshold_mask = last_threshold_mask.to(zo.dtype)
                            tl.store(zo_ptr, last_threshold_mask, boundary_check=(1,))
                            similarity = tl.sum(similarity, axis=1, keep_dims=True)
                            s = similarity + s
                s_ptr = tl.make_block_ptr(
                    base=similarity_cos_ptr
                    + batch_idx * stride_sz
                    + head_idx * stride_sh
                    + m_block_idx * stride_sb
                    + layer_id * stride_sy,
                    shape=(block_size, 1),
                    strides=(stride_sl, 1),
                    offsets=(m_block_offset, 0),
                    block_shape=(BLOCK_M, 1),
                    order=(1, 0),
                )
                tl.store(s_ptr, s.to(raw_dtype), boundary_check=(0,))


def raw_similarity_score(
    key_cache: torch.Tensor,
    block_table: torch.Tensor,
    threshold: float = 0.5,
    temperature: float = 1.0,
    return_logits: bool = False,
):
    """
    Calculate cos similarity score between key cache.
    Args:
        key_cache: (num_layers, num_kvcache_blocks, block_size, num_kv_heads, head_dim)
        block_table: (batch_size, max_num_blocks_per_seq)
        last_block: (batch_size)
        retain_ratio: float
        threshold: float
    Returns:
        similarity_score: (batch_size, num_kv_heads, max_num_blocks_per_seq, block_size)
    """
    BLOCK_M = 16
    BLOCK_N = 64

    num_layers, _, block_size, num_kv_heads, head_dim = key_cache.shape
    batch_size, max_num_blocks_per_seq = block_table.shape

    norm = key_norm(key_cache, block_table)

    similarity_cos = torch.full(
        (num_layers, batch_size, num_kv_heads, max_num_blocks_per_seq, block_size),
        -float("inf"),
        device=key_cache.device,
        dtype=key_cache.dtype,
    )

    zero_out = torch.full(
        (num_layers, batch_size, num_kv_heads, max_num_blocks_per_seq, block_size),
        False,
        device=key_cache.device,
        dtype=torch.bool,
    )
    grid = (batch_size * num_kv_heads, num_layers)

    raw_similarity_score_kernel[grid](
        key_cache,
        similarity_cos,
        norm,
        zero_out,
        block_table,
        **_strides(key_cache, "ky", "kb", "kl", "kh", "kd"),
        **_strides(similarity_cos, "sy", "sz", "sh", "sb", "sl"),
        **_strides(norm, "ny", "nz", "nh", "nb", "nl"),
        **_strides(zero_out, "zy", "zz", "zh", "zb", "zl"),
        **_strides(block_table, "tz", "tb"),
        max_num_blocks_per_seq=max_num_blocks_per_seq,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        block_size=block_size,
        batch_size=batch_size,
        head_dim=head_dim,
        threshold=threshold,
    )
    del zero_out
    del norm

    similarity_cos = similarity_cos.view(num_layers, batch_size, num_kv_heads, -1)
    if return_logits:
        logits = similarity_cos.clone()

    seq_length = (block_table != -1).sum(dim=-1) * block_size
    similarity_cos = similarity_cos.div_(
        temperature * seq_length.unsqueeze(-1).unsqueeze(-1)
    )
    dtype = similarity_cos.dtype
    similarity_cos = similarity_cos.float()
    similarity_cos = similarity_cos - similarity_cos.max(dim=-1, keepdim=True).values
    similarity_cos = similarity_cos.softmax(dim=-1)
    similarity_cos = similarity_cos.to(dtype)

    similarity_cos = similarity_cos.reshape(
        num_layers, batch_size, num_kv_heads, max_num_blocks_per_seq, block_size
    )

    if return_logits:
        return logits, similarity_cos

    return similarity_cos
