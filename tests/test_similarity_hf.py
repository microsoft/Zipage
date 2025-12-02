import os

# os.environ["TRITON_INTERPRET"] = "1"

import torch
import triton
import triton.language as tl
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from zipvllm.kernel.utils import _strides


@triton.jit
def similarity_score_kernel(
    key_states_ptr,
    similarity_cos_ptr,
    key_norm_ptr,
    num_pad_ptr,
    zero_out_ptr,
    stride_kb,
    stride_kh,
    stride_kl,
    stride_kd,
    stride_sb,
    stride_sh,
    stride_sl,
    stride_nb,
    stride_nh,
    stride_nl,
    stride_zb,
    stride_zh,
    stride_zl,
    threshold: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    batch_size: tl.constexpr,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_idx = pid % batch_size
    head_idx = pid // batch_size

    num_pad = tl.load(num_pad_ptr + batch_idx)

    for m_offset in range(seq_len - BLOCK_M, num_pad - BLOCK_M, -BLOCK_M):

        km_ptr = tl.make_block_ptr(
            base=key_states_ptr + batch_idx * stride_kb + head_idx * stride_kh,
            shape=(seq_len, head_dim),
            strides=(stride_kl, stride_kd),
            offsets=(m_offset, 0),
            block_shape=(BLOCK_M, head_dim),
            order=(1, 0),
        )
        # (m, d)
        km = tl.load(km_ptr, boundary_check=(0,))
        km_norm_ptr = tl.make_block_ptr(
            base=key_norm_ptr + batch_idx * stride_nb + head_idx * stride_nh,
            shape=(seq_len, 1),
            strides=(stride_nl, 1),
            offsets=(m_offset, 0),
            block_shape=(BLOCK_M, 1),
            order=(1, 0),
        )
        # (m,1)
        km_norm = tl.load(km_norm_ptr, boundary_check=(0,))
        km = km / (km_norm + 1e-6)
        similarity_ptr = tl.make_block_ptr(
            base=similarity_cos_ptr + batch_idx * stride_sb + head_idx * stride_sh,
            shape=(seq_len, 1),
            strides=(stride_sl, 1),
            offsets=(m_offset, 0),
            block_shape=(BLOCK_M, 1),
            order=(1, 0),
        )
        s = tl.load(similarity_ptr, boundary_check=(0,))
        raw_dtype = s.dtype
        s = s.to(tl.float32)
        for n_offset in range(num_pad, seq_len, BLOCK_N):
            kn_ptr = tl.make_block_ptr(
                base=key_states_ptr + batch_idx * stride_kb + head_idx * stride_kh,
                shape=(head_dim, seq_len),
                strides=(stride_kd, stride_kl),
                offsets=(0, n_offset),
                block_shape=(head_dim, BLOCK_N),
                order=(0, 1),
            )
            # (d, n)
            kn = tl.load(kn_ptr, boundary_check=(1,))
            kn_norm_ptr = tl.make_block_ptr(
                base=key_norm_ptr + batch_idx * stride_nb + head_idx * stride_nh,
                shape=(1, seq_len),
                strides=(1, stride_nl),
                offsets=(0, n_offset),
                block_shape=(1, BLOCK_N),
                order=(0, 1),
            )
            kn_norm = tl.load(kn_norm_ptr, boundary_check=(1,))
            kn = kn / (kn_norm + 1e-6)
            # (m, n)
            similarity = tl.dot(km, kn)
            # zero out the similarity of the same key
            m_indices = m_offset + tl.arange(0, BLOCK_M)[:, None]
            n_indices = n_offset + tl.arange(0, BLOCK_N)[None, :]
            same_key_mask = m_indices == n_indices
            similarity = tl.where(same_key_mask, 0.0, similarity)
            # zero out the last token with high similarity
            zo_ptr = tl.make_block_ptr(
                base=zero_out_ptr + batch_idx * stride_zb + head_idx * stride_zh,
                shape=(1, seq_len),
                strides=(1, stride_zl),
                offsets=(0, n_offset),
                block_shape=(1, BLOCK_N),
                order=(0, 1),
            )
            zo = tl.load(zo_ptr, boundary_check=(1,))
            threshold_mask = ((similarity > threshold) & (~zo)).to(tl.int1)
            row_indices = tl.arange(0, BLOCK_M)[:, None]
            max_row_per_col = tl.max(
                tl.where(
                    threshold_mask,
                    row_indices,
                    -1,
                ),
                axis=0,
                keep_dims=True,
            )
            last_threshold_mask = (row_indices == max_row_per_col) & threshold_mask
            similarity = tl.where(last_threshold_mask, 0.0, similarity)
            last_threshold_mask = tl.max(last_threshold_mask, axis=0, keep_dims=True)
            last_threshold_mask = last_threshold_mask | zo
            last_threshold_mask = last_threshold_mask.to(zo.dtype)
            tl.store(zo_ptr, last_threshold_mask, boundary_check=(1,))
            # reduce cosine similarity
            similarity = tl.sum(similarity, axis=1, keep_dims=True)
            s = similarity + s
        tl.store(similarity_ptr, s.to(raw_dtype), boundary_check=(0,))


def cal_similarity(
    key_states: torch.Tensor,
    attention_mask: torch.Tensor,
    threshold=0.1,
    temperature=1.0,
    debug=False,
):
    """
    calculate cosine similarity score between key states
    the last token with high similarity (similarity > threshold) will be masked out
    the similarity score of the same key will also be masked out
    Args:
        key_states: (batch_size, num_kv_heads, seq_len, head_dim)
        attention_mask: (batch_size,  seq_len)
        threshold: float
    Returns:
        similarity_score: (batch_size, num_heads, seq_len)
    """
    BLOCK_M = 16
    BLOCK_N = 64

    batch_size, num_heads, seq_len, head_dim = key_states.shape

    key_norm = key_states.norm(dim=-1)
    similarity_cos = torch.zeros(
        (batch_size, num_heads, seq_len),
        dtype=key_states.dtype,
        device=key_states.device,
    )
    # tag for whether the similarity score of the same key has been masked out
    zero_out = torch.full(
        (batch_size, num_heads, seq_len),
        False,
        device=key_states.device,
        dtype=torch.bool,
    )
    num_pad = (attention_mask == 0).sum(dim=-1).to(torch.int32)

    grid = (batch_size * num_heads,)
    similarity_score_kernel[grid](
        key_states,
        similarity_cos,
        key_norm,
        num_pad,
        zero_out,
        **_strides(key_states, "kb", "kh", "kl", "kd"),
        **_strides(similarity_cos, "sb", "sh", "sl"),
        **_strides(key_norm, "nb", "nh", "nl"),
        **_strides(zero_out, "zb", "zh", "zl"),
        threshold=threshold,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        batch_size=batch_size,
        seq_len=seq_len,
        head_dim=head_dim,
    )

    seq_len = attention_mask.sum(dim=-1, keepdim=True).unsqueeze(-1)
    if debug:
        logits = similarity_cos.clone()
    similarity_cos.div_(seq_len * temperature)

    similarity_cos = torch.softmax(
        similarity_cos - similarity_cos.max(dim=-1, keepdim=True).values, dim=-1
    )

    if debug:
        return logits, similarity_cos
    return similarity_cos


def test():
    batch_size = 2
    num_heads = 12
    seq_len = 1024
    head_dim = 128
    key_cache = torch.randn(
        batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float16
    )
    attention_mask = torch.tensor(
        [[0] * 32 + [1] * (seq_len - 32), [1] * seq_len],
        device="cuda",
    )
    similarity_score = cal_similarity(key_cache, attention_mask)
    time_start = time.time()
    cal_similarity(key_cache, attention_mask)
    time_end = time.time()
    print(similarity_score)
    print(f"Triton time: {time_end - time_start}")


if __name__ == "__main__":
    test()
