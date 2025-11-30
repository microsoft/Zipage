import os

# os.environ["TRITON_INTERPRET"] = "1"

import torch
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from zipvllm.kernel.light_similarity_score import light_similarity_score


def cal_similarity_ref(key_cache, block_table, last_block, threshold=0.5):
    num_kvcache_blocks, block_size, num_kv_heads, head_dim = key_cache.shape
    batch_size, max_num_blocks_per_seq = block_table.shape

    # (batch_size, num_kv_heads, max_num_blocks_per_seq , block_size，head_dim)

    key_states = torch.zeros(
        batch_size,
        num_kv_heads,
        max_num_blocks_per_seq,
        block_size,
        head_dim,
        device=key_cache.device,
        dtype=key_cache.dtype,
    )
    for z in range(batch_size):
        for h in range(num_kv_heads):
            for m in range(max_num_blocks_per_seq):
                if block_table[z, m] != -1:
                    if block_table[z, m] < 0:
                        block_id = -block_table[z, m] - 2
                    else:
                        block_id = block_table[z, m]
                    key_states[z, h, m] = key_cache[block_id, :, h, :].view(
                        block_size, head_dim
                    )

    last_block_key = torch.zeros(
        batch_size,
        num_kv_heads,
        block_size,
        head_dim,
        device=key_cache.device,
        dtype=key_cache.dtype,
    )

    for z in range(batch_size):
        for h in range(num_kv_heads):
            block_id = last_block[z]
            last_block_key[z, h] = key_cache[block_id, :, h, :].view(
                block_size, head_dim
            )
    key_states = key_states.view(batch_size, num_kv_heads, -1, head_dim)
    key_norm = key_states.norm(dim=-1, keepdim=True)
    key_norm = torch.clamp(key_norm, min=1e-6)
    key_states = key_states / (key_norm)

    last_block_norm = last_block_key.norm(dim=-1, keepdim=True)
    last_block_norm = torch.clamp(last_block_norm, min=1e-6)
    last_block_key = last_block_key / (last_block_norm)

    # (batch_size, num_kv_heads, seq_len, block_size)
    similarity = key_states @ last_block_key.transpose(-2, -1)

    similarity = similarity.sum(dim=-1)
    seq_length = (block_table != -1).sum(dim=-1) * block_size
    # (batch_size, max_num_blocks_per_seq)
    valid_mask = block_table == -1
    # mask out columns and rows beyond seq_len
    valid_mask = (
        valid_mask.unsqueeze(1)
        .unsqueeze(-1)
        .expand(batch_size, num_kv_heads, max_num_blocks_per_seq, block_size)
    )
    valid_mask = valid_mask.reshape(batch_size, num_kv_heads, -1)
    similarity.masked_fill_(
        valid_mask,
        float("-inf"),
    )
    logits = similarity.clone()

    similarity = similarity.div_(seq_length.unsqueeze(-1).unsqueeze(-1))
    similarity = similarity.softmax(dim=-1)
    similarity = similarity.reshape(
        batch_size, num_kv_heads, max_num_blocks_per_seq, block_size
    )
    return similarity, logits


def test():
    num_kvcache_blocks = 16
    block_size = 256
    num_kv_heads = 2
    head_dim = 128
    key_cache = torch.randn(
        num_kvcache_blocks,
        block_size,
        num_kv_heads,
        head_dim,
        device="cuda",
        dtype=torch.float16,
    )
    block_table = torch.tensor(
        [[0, 1, -3], [4, -7, -1]], device="cuda", dtype=torch.int32
    )
    last_block = torch.tensor([2, 6], device="cuda", dtype=torch.int32)
    similarity_score, logits = light_similarity_score(
        key_cache, block_table, last_block, debug=True
    )

    similarity_score_ref, logits_ref = cal_similarity_ref(
        key_cache, block_table, last_block
    )

    time_start = time.time()
    similarity_score, logits = light_similarity_score(
        key_cache, block_table, last_block, debug=True
    )
    time_end = time.time()
    logits=logits.reshape(2,num_kv_heads,-1)

    diff = (similarity_score - similarity_score_ref).abs()
    print("sum diff:", diff.sum().item())
    print("max diff:", diff.max().item())
    print(f"Triton time: {time_end - time_start}")
    print(similarity_score.shape)
    print(similarity_score.min(), similarity_score.max())


if __name__ == "__main__":
    test()
