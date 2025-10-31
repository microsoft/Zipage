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

from zipvllm.kernel.store_query_cache import store_query_cache


def test():
    total_n = 128
    max_num_seqs = 16
    query_cache_len = 16
    num_heads = 12
    head_dim = 64
    query = torch.arange(total_n, device="cuda", dtype=torch.float16).reshape(
        total_n, 1, 1
    )
    query = query.repeat(1, num_heads, head_dim).contiguous()

    query_cache = torch.zeros(
        max_num_seqs,
        query_cache_len,
        num_heads,
        head_dim,
        device="cuda",
        dtype=torch.float16,
    )
    slot_mapping = torch.tensor(
        [
            [3, 3, 0],
            [4, 3, 1],
            [5, 3, 2],
            [6, 3, 3],
            [7, 3, 4],
            [8, 3, 5],
            [9, 3, 6],
            [10, 3, 7],
            [11, 3, 8],
            [12, 3, 9],
            [13, 3, 10],
            [14, 3, 11],
            [15, 3, 12],
            [64, 0, 0],
            [65, 0, 1],
            [66, 0, 2],
            [67, 0, 3],
            [68, 0, 4],
            [69, 0, 5],
            [70, 0, 6],
            [71, 0, 7],
            [72, 0, 8],
            [73, 0, 9],
        ],
        device="cuda",
        dtype=torch.int32,
    )
    store_query_cache(query, query_cache, slot_mapping)

    start_time = time.time()
    store_query_cache(query, query_cache, slot_mapping)
    end_time = time.time()
    # position
    print(query_cache[3, :, 0, 0])
    print(query_cache[0, :, 0, 0])
    # completion
    print(query_cache[3, 0, :, :])
    print(query_cache[0, 0, :, :])
    print(f"Time taken: {end_time - start_time} seconds")

if __name__ == "__main__":
    test()
