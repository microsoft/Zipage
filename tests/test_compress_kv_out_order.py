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

from zipvllm.kernel.compress_kv_out_order import compress_kv_out_order
from zipvllm.kernel.compress_score_out_order import compress_score_out_order


def topk_mask(input: torch.Tensor, k: int, dim: int = -1):
    """

    calculate if the last dimension of input is top-k, return a bool mask with the same shape
    top-k is True, others are False
    """
    topk_indices = torch.topk(input, k, dim=dim)[1]
    mask = torch.zeros_like(input, dtype=torch.bool)
    mask.scatter_(dim, topk_indices, True)
    return mask


def get_compress_slot_indices(
    flag: torch.Tensor, block_table: torch.Tensor, kept_blocks: int
):
    block_valid_mask = (block_table[:, kept_blocks:][:, None, :, None]) != -1

    save_indices = (
        (~flag[:, :, :kept_blocks, :]).nonzero(as_tuple=False).contiguous()
    )
    load_indices = (
        (flag[:, :, kept_blocks:, :] & block_valid_mask)
        .nonzero(as_tuple=False)
        .contiguous()
    )
    if load_indices.numel() > 0:
        load_indices[:, 2].add_(kept_blocks)

    return save_indices, load_indices


def test():
    num_kvcache_blocks = 8
    block_size = 256
    num_kv_heads = 2
    head_dim = 128
    batch_size = 2
    kept_blocks = 2
    max_num_blocks_per_seq = 4

    score = torch.randn(
        batch_size,
        num_kv_heads,
        max_num_blocks_per_seq,
        block_size,
        device="cuda",
        dtype=torch.float16,
    )
    block_table = torch.tensor(
        [[0, 1, 2, -5], [5, 6, -9, -1]], device="cuda", dtype=torch.int32
    )
    last_block = torch.tensor([3, 2], device="cuda", dtype=torch.int32)
    mask = block_table == -1
    mask = mask.unsqueeze(1).unsqueeze(-1).repeat(1, num_kv_heads, 1, block_size)
    score[mask] = -1e3

    score = score.reshape(batch_size, num_kv_heads, -1)
    total_slots = block_size * kept_blocks
    flag = topk_mask(score, total_slots)
    flag = flag.reshape(batch_size, num_kv_heads, max_num_blocks_per_seq, block_size)

    save_indices, load_indices = get_compress_slot_indices(
        flag, block_table, kept_blocks
    )
    
    assert save_indices.shape == load_indices.shape
    assert torch.all(save_indices[:, :2] == load_indices[:, :2])

    for _ in range(2):
        # loop 2 times to warm up
        key = torch.arange(
            num_kvcache_blocks * block_size, device="cuda", dtype=torch.float16
        ).reshape(num_kvcache_blocks, block_size, 1, 1)
        key = key.repeat(1, 1, num_kv_heads, head_dim).contiguous()
        value = torch.arange(
            num_kvcache_blocks * block_size, device="cuda", dtype=torch.float16
        ).reshape(num_kvcache_blocks, block_size, 1, 1)
        value = value.repeat(1, 1, num_kv_heads, head_dim).contiguous()

        start_time = time.time()
        compress_kv_out_order(key, value, save_indices, load_indices, block_table)
        end_time = time.time()

    score_cache = torch.arange(
        num_kvcache_blocks * block_size, device="cuda", dtype=torch.float16
    ).reshape(num_kvcache_blocks, block_size, 1)
    score_cache = score_cache.repeat(1, 1, num_kv_heads).contiguous()
    start_time = time.time()
    compress_score_out_order(score_cache, save_indices, load_indices, block_table)
    end_time = time.time()

    print("key: ", key[0, :, 0, 0])
    print("score: ", score_cache[0, :, 0])
    print("key: ", key[1, :, 0, 0])
    print("score: ", score_cache[1, :, 0])
    print("key: ", key[5, :, 0, 0])
    print("score: ", score_cache[5, :, 0])
    print("key: ", key[6, :, 0, 0])
    print("score: ", score_cache[6, :, 0])
    print("compress kv out order time: ", end_time - start_time)


if __name__ == "__main__":
    test()
