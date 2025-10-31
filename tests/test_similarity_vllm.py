import os

os.environ["TRITON_INTERPRET"] = "1"

import torch
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent))
from zipvllm.kernel.utils import _strides

from zipvllm.kernel.raw_similarity_score import raw_similarity_score


def cal_similarity_ref(key_cache, block_table, seq_length, threshold=0.5):
    num_kvcache_blocks, block_size, num_kv_heads, head_dim = key_cache.shape
    batch_size, max_num_blocks_per_seq = block_table.shape

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
                if block_table[z, m] >= 0:
                    key_states[z, h, m] = key_cache[block_table[z, m], :, h, :].view(
                        block_size, head_dim
                    )
    key_states = key_states.view(batch_size, num_kv_heads, -1, head_dim)
    key_norm = key_states.norm(dim=-1, keepdim=True)
    key_states = key_states / (key_norm + 1e-6)
    similarity = key_states @ key_states.transpose(-2, -1)
    col_indices = torch.arange(
        0, block_size * max_num_blocks_per_seq, device=key_cache.device
    )[None, :]
    row_indices = torch.arange(
        0, block_size * max_num_blocks_per_seq, device=key_cache.device
    )[:, None]
    same_key_mask = row_indices == col_indices
    similarity = torch.where(same_key_mask, 0.0, similarity)
    max_val, max_idx = similarity.max(dim=-1, keepdim=True)
    similarity = torch.where(
        (similarity == max_val) & (similarity > threshold), 0.0, similarity
    )
    similarity = similarity.sum(dim=-2)
    for z in range(batch_size):
        seq_len = seq_length[z].item()
        valid_mask = (
            torch.arange(block_size * max_num_blocks_per_seq, device=key_cache.device)
            < seq_len
        )
        # mask out columns and rows beyond seq_len
        similarity[z, ~valid_mask.unsqueeze(0).expand(num_kv_heads, -1)] = float("-inf")

    similarity = similarity.div_(seq_length.unsqueeze(-1).unsqueeze(-1))
    similarity = similarity.softmax(dim=-1)
    return similarity.reshape(
        batch_size, num_kv_heads, max_num_blocks_per_seq, block_size
    )


def test():
    num_kvcache_blocks = 16
    block_size = 256
    num_kv_heads = 2
    head_dim = 128
    batch_size = 2
    max_num_blocks_per_seq = 4
    key_cache = torch.randn(
        num_kvcache_blocks,
        block_size,
        num_kv_heads,
        head_dim,
        device="cuda",
        dtype=torch.float16,
    )
    block_table = torch.tensor(
        [[0, 1, 2, 3], [4, 5, 6, -1]], device="cuda", dtype=torch.int32
    )
    seq_length = torch.tensor([256 * 4, 256 * 3], device="cuda", dtype=torch.int32)
    similarity_score = raw_similarity_score(key_cache, block_table)

    similarity_score_ref = cal_similarity_ref(key_cache, block_table, seq_length, 0.5)

    time_start = time.time()
    similarity_score = raw_similarity_score(key_cache, block_table)
    time_end = time.time()

    diff = (similarity_score - similarity_score_ref).abs()
    print("sum diff:", diff.sum().item())
    print("max diff:", diff.max().item())
    print(f"Triton time: {time_end - time_start}")
    print(similarity_score.shape)
    print(similarity_score.min(), similarity_score.max())


if __name__ == "__main__":
    test()
