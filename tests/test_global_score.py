import os

# os.environ["TRITON_INTERPRET"] = "1"
import torch
import numpy as np
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from zipage.kernel.attention_score import attention_score
from zipage.kernel.global_score import global_score


def load_score_cache(score_cache, block_table):
    num_layers, num_blocks, block_size, num_kv_heads = score_cache.shape
    batch_size, max_num_blocks_per_seq = block_table.shape

    score = torch.zeros(
        num_layers,
        batch_size,
        max_num_blocks_per_seq,
        block_size,
        num_kv_heads,
        device=score_cache.device,
        dtype=score_cache.dtype,
    )
    for l in range(num_layers):
        for i in range(batch_size):
            for j in range(max_num_blocks_per_seq):
                block_id = block_table[i, j]
                if not block_id == -1:
                    if block_id < 0:
                        block_id = -block_id - 2
                    score[l, i, j, :, :] = score_cache[l, block_id, :, :]
    return score.permute(0, 1, 4, 2, 3)


def test():
    dim = 128
    num_blocks = 32
    block_size = 256
    num_kv_heads = 2
    num_attention_heads = 12
    max_num_seqs = 256
    query_cache_len = 32
    num_layers = 2
    dtype = torch.float16
    k_cache = (
        torch.randn(
            num_layers,
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
            num_layers,
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
        num_layers, num_blocks, block_size, num_kv_heads, device="cuda", dtype=dtype
    )

    compressed = torch.tensor([False, True], device="cuda", dtype=torch.bool)

    # triton warmup
    scores = attention_score(k_cache, query_cache, seq_idx, block_table)
    scores = global_score(scores, score_cache, block_table, compressed).reshape(
        num_layers, 2, num_kv_heads, -1
    )
    loaded_score = load_score_cache(score_cache, block_table).reshape(num_layers, 2, num_kv_heads, -1)
    diff = (scores - loaded_score).abs().sum()
    print(f"max diff: {diff.max().item()}")
    print(f"sum diff: {diff.sum().item()}")
    print(f"mean diff: {diff.mean().item()}")


    


if __name__ == "__main__":
    test()
