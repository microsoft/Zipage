import os

# os.environ["TRITON_INTERPRET"] = "1"
import torch
import numpy as np
import sys
import time
from pathlib import Path

torch.set_printoptions(precision=8, sci_mode=False)
np.set_printoptions(precision=8, suppress=True)

sys.path.insert(0, str(Path(__file__).parent.parent))

from zipvllm.kernel.attention_score import attention_score


def attention_score_ref(key_cache, query_cache, seq_idx, block_table, scale=None):
    num_layers, num_kvcache_blocks, block_size, num_kv_heads, head_dim = key_cache.shape
    _, _, query_cache_len, num_attention_heads, _ = query_cache.shape
    batch_size, max_num_blocks_per_seq = block_table.shape
    if scale is None:
        scale = head_dim ** (-0.5)
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
        dtype=key_cache.dtype,
        device=key_cache.device,
    )
    for l in range(num_layers):
        for z in range(batch_size):
            for h in range(num_attention_heads):
                # (query_cache_len, head_dim)
                q = query_cache[l, seq_idx[z].item(), :, h, :]
                for m in range(max_num_blocks_per_seq):
                    if block_table[z, m] != -1:
                        if block_table[z, m] < 0:
                            block_id = -block_table[z, m] - 2
                            last_block = True
                        else:
                            block_id = block_table[z, m]
                            last_block = False
                        hk = h // (num_attention_heads // num_kv_heads)
                        # (block_size, head_dim)
                        k = key_cache[l, block_id, :, hk, :]
                        qk = q @ k.transpose(-2, -1) * scale
                        if last_block:
                            col_index = torch.arange(block_size)[None, :].to(
                                key_cache.device
                            )
                            raw_index = torch.arange(
                                block_size - query_cache_len, block_size
                            )[:, None].to(key_cache.device)
                            mask = col_index > raw_index
                            qk = qk.masked_fill(mask, -float("inf"))
                        qk_buffer[l, z, h, :, m, :] = qk
    qk_buffer = qk_buffer.reshape(
        num_layers, batch_size, num_attention_heads, query_cache_len, -1
    )
    score = torch.softmax(
        qk_buffer - qk_buffer.max(dim=-1, keepdim=True).values, dim=-1
    )
    score = score.reshape(
        num_layers,
        batch_size,
        num_kv_heads,
        num_attention_heads // num_kv_heads,
        query_cache_len,
        max_num_blocks_per_seq,
        block_size,
    )
    score = score.max(dim=3).values
    score = score.mean(dim=3)
    return score, qk_buffer


def test():
    num_kvcache_blocks = 20
    block_size = 256
    num_kv_heads = 2
    head_dim = 128
    max_num_seqs = 10
    query_cache_len = 16
    num_attention_heads = 16
    layer = 2
    key_cache = torch.randn(
        layer,
        num_kvcache_blocks,
        block_size,
        num_kv_heads,
        head_dim,
        device="cuda",
        dtype=torch.float16,
    )
    query_cache = torch.randn(
        layer,
        max_num_seqs,
        query_cache_len,
        num_attention_heads,
        head_dim,
        device="cuda",
        dtype=torch.float16,
    )
    block_table = torch.tensor(
        [[0, 1, 2, -5], [5, 6, -9, -1]], device="cuda", dtype=torch.int32
    )
    seq_idx = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    score, logits = attention_score(
        key_cache, query_cache, seq_idx, block_table, return_logits=True
    )
    score_ref, logits_ref = attention_score_ref(
        key_cache, query_cache, seq_idx, block_table
    )

    logits[logits == float("-inf")] = 0
    logits_ref[logits_ref == float("-inf")] = 0

    diff = (logits - logits_ref).abs()
    print("sum diff:", diff.sum())
    print("max diff:", diff.max())
    print("mean diff:", diff.mean())


if __name__ == "__main__":
    test()
