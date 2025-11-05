import os

# os.environ["TRITON_INTERPRET"] = "1"
import torch
import numpy as np
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from zipvllm.kernel.attention_score import attention_score, get_padded_headsize
from zipvllm.kernel.global_score import global_score
from zipvllm.kernel.compress_score import compress_score


def calculate_attention_scores(
    k_cache, query_cache, seq_idx, block_table, reduced="mean", gamma=0.99
):
    num_blocks, block_size, num_kv_heads, _ = k_cache.shape
    _, query_cache_len, num_attention_heads, _ = query_cache.shape
    batch_size = seq_idx.shape[0]
    max_num_blocks_per_seq = block_table.shape[1]
    qk_buffer = torch.full(
        (
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
    softmax_scale = query_cache.shape[-1] ** (-0.5)
    for i in range(batch_size):
        seq_id = seq_idx[i]
        q = query_cache[
            seq_id, :, :, :
        ]  # (query_cache_len, num_attention_heads, head_dim)
        q = q.reshape(
            query_cache_len, num_kv_heads, num_attention_heads // num_kv_heads, -1
        )
        # (num_kv_heads, num_groups, query_cache_len, head_dim)
        q = q.permute(1, 2, 0, 3)
        for j in range(max_num_blocks_per_seq):
            block_id = block_table[i, j]
            if not block_id == -1:
                if block_id < 0:
                    block_id = -block_id - 2
                # (block_size, num_kv_heads, head_dim)
                k = k_cache[block_id, :, :, :]
                k = k.unsqueeze(2)
                # (num_kv_heads, 1, head_dim, block_size)
                k = k.permute(1, 2, 3, 0)
                # (num_kv_heads, num_groups, query_cache_len, block_size)
                qk = q @ k
                qk = qk * softmax_scale
                qk_buffer[i, :, :, j, :] = qk.reshape(
                    num_attention_heads, query_cache_len, block_size
                )
    qk_buffer = qk_buffer.reshape(batch_size, num_attention_heads, query_cache_len, -1)

    score = torch.softmax(
        qk_buffer - qk_buffer.max(dim=-1, keepdim=True).values, dim=-1
    )
    del qk_buffer
    score = score.view(
        batch_size,
        num_kv_heads,
        num_attention_heads // num_kv_heads,
        query_cache_len,
        -1,
    )
    reduced_scores = torch.max(score, dim=2).values
    del score
    reduced_scores = reduced_scores.reshape(
        batch_size, num_kv_heads, query_cache_len, max_num_blocks_per_seq, block_size
    )
    if reduced == "mean":
        reduced_scores = reduced_scores.mean(dim=2)
    elif reduced == "max":
        reduced_scores = reduced_scores.max(dim=2).values
    elif reduced == "weighted_mean":
        weights = [gamma]
        for i in range(query_cache_len - 1):
            weights.append(weights[-1] * gamma)
        weights = weights[::-1]
        weights = torch.tensor(
            weights, device=reduced_scores.device, dtype=reduced_scores.dtype
        )
        reduced_scores = (reduced_scores * weights.view(1, 1, -1, 1, 1)).mean(dim=2)
    elif reduced == "weighted_max":
        weights = [gamma]
        for i in range(query_cache_len - 1):
            weights.append(weights[-1] * gamma)
        weights = weights[::-1]
        weights = torch.tensor(
            weights, device=reduced_scores.device, dtype=reduced_scores.dtype
        )
        reduced_scores = (
            (reduced_scores * weights.view(1, 1, -1, 1, 1)).max(dim=2).values
        )
    else:
        raise ValueError(f"Invalid reduction method: {reduced}")

    return reduced_scores


def load_score_cache(score_cache, block_table):
    num_blocks, block_size, num_kv_heads = score_cache.shape
    batch_size, max_num_blocks_per_seq = block_table.shape

    score = torch.zeros(
        batch_size,
        max_num_blocks_per_seq,
        block_size,
        num_kv_heads,
        device=score_cache.device,
        dtype=score_cache.dtype,
    )
    for i in range(batch_size):
        for j in range(max_num_blocks_per_seq):
            block_id = block_table[i, j]
            if not block_id == -1:
                if block_id < 0:
                    block_id = -block_id - 2
                score[i, j, :, :] = score_cache[block_id, :, :]
    return score.permute(0, 3, 1, 2)


def test():
    dim = 128
    num_blocks = 32
    block_size = 256
    num_kv_heads = 2
    num_attention_heads = 12
    max_num_seqs = 256
    query_cache_len = 32
    dtype = torch.float16
    k_cache = (
        torch.randn(
            num_blocks,
            block_size,
            num_kv_heads,
            dim,
            dtype=dtype,
            device="cuda",
        )
        * 2
    )
    query_cache = (
        torch.randn(
            max_num_seqs,
            query_cache_len,
            num_attention_heads,
            dim,
            dtype=dtype,
            device="cuda",
        )
        * 2
    )

    seq_idx = torch.arange(2, dtype=torch.int32, device="cuda")
    block_table = torch.tensor(
        [[0, 1, -4], [8, -9, -1]],
        dtype=torch.int32,
        device="cuda",
    )
    score_cache = torch.zeros(
        num_blocks, block_size, num_kv_heads, device="cuda", dtype=dtype
    )

    compressed = torch.tensor([True, False], device="cuda", dtype=torch.bool)

    # triton warmup
    result = attention_score(k_cache, query_cache, seq_idx, block_table)
    global_score(result, score_cache, block_table, compressed)
    score_cache.zero_()

    # torch and loop
    time_start = time.time()
    ref_result = calculate_attention_scores(
        k_cache, query_cache, seq_idx, block_table, reduced="mean", gamma=0.99
    )
    time_end = time.time()
    print(f"Torch time: {time_end - time_start}")
    # triton
    time_start = time.time()
    result = attention_score(
        k_cache, query_cache, seq_idx, block_table, reduced="mean", gamma=0.99
    )
    time_end = time.time()
    print(f"Triton time: {time_end - time_start}")
    # diff
    diff = (result - ref_result).abs()
    print("sum diff:", diff.sum())
    print("max diff:", diff.max())

    # store score
    time_start = time.time()
    maxed_score = global_score(result, score_cache, block_table, compressed).reshape(
        2, num_kv_heads, -1
    )
    time_end = time.time()
    loaded_score = load_score_cache(score_cache, block_table)
    result = result.reshape(2, num_kv_heads, -1)
    ref_result = ref_result.reshape(2, num_kv_heads, -1)
    loaded_score = loaded_score.reshape(2, num_kv_heads, -1)
    print(f"Store score time: {time_end - time_start}")
    print("max score - saved score diff:")
    print("sum diff:", (maxed_score - loaded_score).abs().sum())
    print("max diff:", (maxed_score - loaded_score).abs().max())
    print("saved score - ref score diff:")
    print("sum diff:", (loaded_score - ref_result).abs().sum())
    print("max diff:", (loaded_score - ref_result).abs().max())


if __name__ == "__main__":
    test()
