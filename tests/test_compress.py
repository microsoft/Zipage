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

from zipvllm.kernel.compress_kv import compress_kv
from zipvllm.kernel.compress_score import compress_score
from zipvllm.kernel.utils import topk_mask


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
        [[0, 1, 2, 3], [5, 6, 7, -5]], device="cuda", dtype=torch.int32
    )
    mask = block_table < 0
    mask = mask.unsqueeze(1).unsqueeze(-1).repeat(1, num_kv_heads, 1, block_size)
    score[mask] = -1e3

    score = score.reshape(batch_size, num_kv_heads, -1)
    total_slots = block_size * kept_blocks
    flag = topk_mask(score, total_slots)
    flag = flag.reshape(batch_size, num_kv_heads, max_num_blocks_per_seq, block_size)

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
        compress_kv(key, value, flag, block_table)
        end_time = time.time()
    print("key: ", key[:4, :, 0, 0])
    print("key: ", key[4:8, :, 0, 0])
    print("value: ", value[:4, :, 0, 0])
    print("value: ", value[4:8, :, 0, 0])
    print("compress kvtime: ", end_time - start_time)

    for _ in range(2):
        score_cache = torch.arange(
            num_kvcache_blocks * block_size, device="cuda", dtype=torch.float16
        ).reshape(num_kvcache_blocks, block_size, 1)
        score_cache = score_cache.repeat(1, 1, num_kv_heads).contiguous()
        start_time = time.time()
        compress_score(score_cache, flag, block_table)
        end_time = time.time()
    print("compress score time: ", end_time - start_time)
    print("score: ", score_cache[:4, :, 0])
    print("score: ", score_cache[4:8, :, 0])


if __name__ == "__main__":
    test()
