import os

# os.environ["TRITON_INTERPRET"] = "1"

import torch
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent))
from zipage.kernel.utils import _strides

from zipage.kernel.lightning_redudancy_score import lightning_redudancy_score


def cal_redudancy_ref(key_cache, block_table, threshold=0.5, temperature=1.0):
    num_layers, num_kvcache_blocks, block_size, num_kv_heads, head_dim = key_cache.shape
    batch_size, max_num_blocks_per_seq = block_table.shape

    # (batch_size, num_kv_heads, max_num_blocks_per_seq , block_size，head_dim)

    key_states = torch.zeros(
        num_layers,
        batch_size,
        num_kv_heads,
        max_num_blocks_per_seq,
        block_size,
        head_dim,
        device=key_cache.device,
        dtype=key_cache.dtype,
    )
    for l in range(num_layers):
        for z in range(batch_size):
            for h in range(num_kv_heads):
                for m in range(max_num_blocks_per_seq):
                    if block_table[z, m] != -1:
                        if block_table[z, m] < 0:
                            block_id = -block_table[z, m] - 2
                        else:
                            block_id = block_table[z, m]
                        key_states[l, z, h, m] = key_cache[l, block_id, :, h, :].view(
                            block_size, head_dim
                        )

    key_norm = key_states.norm(dim=-1, keepdim=True)
    key_norm = torch.clamp(key_norm, min=1e-6)
    key_states = key_states / (key_norm)

    redudancy = torch.full(
        (num_layers, batch_size, num_kv_heads, max_num_blocks_per_seq, block_size),
        float("-inf"),
        device=key_cache.device,
        dtype=key_cache.dtype,
    )
    for l in range(num_layers):
        for z in range(batch_size):
            for m in range(max_num_blocks_per_seq):
                # (num_kv_heads,block_size, head_dim)
                if block_table[z, m] != -1:
                    k = key_states[l, z, :, m]
                    s = k @ k.transpose(-2, -1)
                    col_indices = torch.arange(0, block_size, device=key_cache.device)[
                        None, :
                    ]
                    row_indices = torch.arange(0, block_size, device=key_cache.device)[
                        :, None
                    ]
                    same_key_mask = row_indices == col_indices
                    s = torch.where(same_key_mask.unsqueeze(0), 0.0, s)

                    mask = s > threshold
                    row_indices = torch.arange(0, block_size, device=key_cache.device)[
                        None, :, None
                    ]
                    max_idx_per_col = torch.where(
                        mask, row_indices, torch.full_like(row_indices, -1)
                    )
                    last_true_idx = max_idx_per_col.max(dim=1, keepdim=True).values
                    last_true_mask = row_indices == last_true_idx
                    s = torch.where(last_true_mask, 0.0, s)
                    s = s.sum(dim=-1)
                    redudancy[l, z, :, m] = s
    logits = redudancy.clone()
    redudancy = redudancy.div_(temperature * block_size)
    redudancy = redudancy.view(num_layers, batch_size, num_kv_heads, -1)
    redudancy = redudancy - redudancy.max(dim=-1, keepdim=True).values
    redudancy = redudancy.softmax(dim=-1)
    redudancy = redudancy.reshape(
        num_layers, batch_size, num_kv_heads, max_num_blocks_per_seq, block_size
    )
    return redudancy, logits


def test():
    num_kvcache_blocks = 40
    block_size = 256
    num_kv_heads = 8
    head_dim = 128
    temperature = 1
    threshold = 0
    num_layers = 4
    key_cache = torch.randn(
        num_layers,
        num_kvcache_blocks,
        block_size,
        num_kv_heads,
        head_dim,
        device="cuda",
        dtype=torch.float32,
    )
    block_table = torch.tensor(
        [[0, 1, 2, -9], [8, 9, -16, -1]],
        device="cuda",
        dtype=torch.int32,
    )

    logits, redudancy_score = lightning_redudancy_score(
        key_cache, block_table, threshold, temperature, return_logits=True
    )

    redudancy_score_ref, logits_ref = cal_redudancy_ref(
        key_cache, block_table, threshold, temperature
    )
    time_start = time.time()
    logits, redudancy_score = lightning_redudancy_score(
        key_cache, block_table, threshold, temperature, return_logits=True
    )
    time_end = time.time()

    logits[logits == float("-inf")] = 0
    logits_ref[logits_ref == float("-inf")] = 0

    diff = (logits - logits_ref).abs()

    flag = diff > 0.005
    # print(flag.tolist())
    print("sum diff:", diff.sum().item())
    print("max diff:", diff.max().item())
    print("mean diff:", diff.mean().item())
    print(f"Triton time: {time_end - time_start}")


if __name__ == "__main__":
    test()
