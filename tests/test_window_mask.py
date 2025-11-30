import os

os.environ["TRITON_INTERPRET"] = "1"
import torch
import numpy as np
import sys
import time
from pathlib import Path

torch.set_printoptions(precision=8, sci_mode=False)
np.set_printoptions(precision=8, suppress=True)

sys.path.insert(0, str(Path(__file__).parent.parent))

from zipvllm.kernel.window_mask import window_mask


def test():
    block_size = 256
    num_kv_heads = 2
    query_cache_len = 16
    max_num_blocks_per_seq = 4
    batch_size = 2
    num_layers = 2
    scores = torch.randn(
        num_layers,
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
    scores = window_mask(scores, block_table, query_cache_len)
    print(scores[0, 0, 0, -1, :])
    print(scores[1, 1, 0, -2, :])


if __name__ == "__main__":
    test()
