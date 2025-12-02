import os

# os.environ["TRITON_INTERPRET"] = "1"

import torch
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent))
from zipvllm.kernel.utils import _strides

from zipvllm.kernel.raw_similarity_score import raw_similarity_score
from tests.test_similarity_hf import cal_similarity


def cal_similarity_ref(key_cache, block_table, threshold=0.5, temperature=1.0):
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
    key_states = key_states.view(num_layers, batch_size, num_kv_heads, -1, head_dim)
    key_norm = key_states.norm(dim=-1, keepdim=True)
    key_norm = torch.clamp(key_norm, min=1e-6)
    key_states = key_states / (key_norm)

    # (num_layers, batch_size, num_kv_heads, seq_len, seq_len)
    similarity = key_states @ key_states.transpose(-2, -1)
    col_indices = torch.arange(
        0, block_size * max_num_blocks_per_seq, device=key_cache.device
    )[None, :]
    row_indices = torch.arange(
        0, block_size * max_num_blocks_per_seq, device=key_cache.device
    )[:, None]
    same_key_mask = row_indices == col_indices
    similarity = torch.where(same_key_mask, 0.0, similarity)
    threshold_mask = (
        similarity > threshold
    )  # (batch_size, num_kv_heads, seq_len, seq_len)
    idx = threshold_mask.flip(-1).float().cumsum(dim=-1)
    last_gt_mask = threshold_mask & (idx.flip(-1) == 1)
    similarity = torch.where(last_gt_mask, 0.0, similarity)
    # (batch_size, num_kv_heads, seq_len)
    similarity = similarity.sum(dim=-2)
    seq_length = (block_table != -1).sum(dim=-1) * block_size

    # (batch_size, max_num_blocks_per_seq)
    valid_mask = block_table == -1
    # mask out columns and rows beyond seq_len
    valid_mask = (
        valid_mask.unsqueeze(1)
        .unsqueeze(-1)
        .unsqueeze(0)
        .expand(
            num_layers, batch_size, num_kv_heads, max_num_blocks_per_seq, block_size
        )
    )
    valid_mask = valid_mask.reshape(num_layers, batch_size, num_kv_heads, -1)
    similarity.masked_fill_(
        valid_mask,
        float("-inf"),
    )
    logits = similarity.clone()

    similarity = similarity.div_(temperature * seq_length.unsqueeze(-1).unsqueeze(-1))
    similarity = similarity - similarity.max(dim=-1, keepdim=True).values
    similarity = similarity.softmax(dim=-1)
    similarity = similarity.reshape(
        num_layers, batch_size, num_kv_heads, max_num_blocks_per_seq, block_size
    )
    return similarity, logits


def cal_similarity_hf(key_cache, block_table, threshold=0.5, temperature=1.0):
    num_layers, num_kvcache_blocks, block_size, num_kv_heads, head_dim = key_cache.shape
    batch_size, max_num_blocks_per_seq = block_table.shape

    padd_block = (block_table == -1).sum(dim=-1)

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
    attention_mask = []
    for z in range(batch_size):
        if padd_block[z].item() == 0:
            attention_mask.append([1] * (max_num_blocks_per_seq * block_size))
        else:
            attention_mask.append(
                [0] * (padd_block[z].item() * block_size)
                + [1] * ((max_num_blocks_per_seq - padd_block[z].item()) * block_size)
            )
    attention_mask = torch.tensor(
        attention_mask, device=key_cache.device, dtype=torch.int64
    )
    for l in range(num_layers):
        for z in range(batch_size):
            for h in range(num_kv_heads):
                pos = padd_block[z].item()
                for m in range(max_num_blocks_per_seq):
                    if block_table[z, m] != -1:
                        if block_table[z, m] < 0:
                            block_id = -block_table[z, m] - 2
                        else:
                            block_id = block_table[z, m]
                        key_states[l, z, h, pos] = key_cache[l, block_id, :, h, :].view(
                            block_size, head_dim
                        )
                        pos += 1
    # (batch_size, num_heads, seq_len)
    key_states = key_states.view(num_layers, batch_size, num_kv_heads, -1, head_dim)

    logits = torch.zeros(
        num_layers,
        batch_size,
        num_kv_heads,
        max_num_blocks_per_seq,
        block_size,
        device=key_cache.device,
        dtype=key_cache.dtype,
    )
    similarity_cos = torch.zeros(
        num_layers,
        batch_size,
        num_kv_heads,
        max_num_blocks_per_seq,
        block_size,
        device=key_cache.device,
        dtype=key_cache.dtype,
    )
    for l in range(num_layers):
        logits_, similarity_cos_ = cal_similarity(
            key_states[l], attention_mask, threshold, temperature, debug=True
        )
        logits[l] = logits_.reshape(
            batch_size, num_kv_heads, max_num_blocks_per_seq, block_size
        )
        similarity_cos[l] = similarity_cos_.reshape(
            batch_size, num_kv_heads, max_num_blocks_per_seq, block_size
        )
    mapped_logits = logits.clone()
    mapped_similarity_cos = similarity_cos.clone().reshape(
        num_layers, batch_size, num_kv_heads, max_num_blocks_per_seq, block_size
    )
    for l in range(num_layers):
        for z in range(batch_size):
            for h in range(num_kv_heads):
                if padd_block[z].item() == 0:
                    continue
                pos = padd_block[z].item()
                for m in range(max_num_blocks_per_seq):
                    if block_table[z, m] != -1:
                        mapped_logits[l, z, h, m, :] = logits[l, z, h, pos, :]
                        mapped_similarity_cos[l, z, h, m, :] = similarity_cos[
                            l, z, h, pos, :
                        ]
                        pos += 1
                    else:
                        mapped_logits[l, z, h, m, :] = 0
                        mapped_similarity_cos[l, z, h, m, :] = 0
    mapped_logits = mapped_logits.view(num_layers, batch_size, num_kv_heads, -1)

    return mapped_logits, mapped_similarity_cos


def test():
    num_kvcache_blocks = 40
    block_size = 256
    num_kv_heads = 8
    head_dim = 128
    temperature = 1
    threshold = 0.5
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
        [[0, 1, 2, 3, 4, 5, 6, -9], [8, 9, 10, 11, 12, 13, -16, -1]],
        device="cuda",
        dtype=torch.int32,
    )
    logits_hf, similarity_score_hf = cal_similarity_hf(
        key_cache, block_table, threshold, temperature
    )
    logits, similarity_score = raw_similarity_score(
        key_cache, block_table, threshold, temperature, return_logits=True
    )

    similarity_score_ref, logits_ref = cal_similarity_ref(
        key_cache, block_table, threshold, temperature
    )
    time_start = time.time()
    logits, similarity_score = raw_similarity_score(
        key_cache, block_table, threshold, temperature, return_logits=True
    )
    time_end = time.time()

    logits[logits == float("-inf")] = 0
    logits_ref[logits_ref == float("-inf")] = 0
    logits_hf[logits_hf == float("-inf")] = 0

    diff = (logits - logits_ref).abs()

    flag = diff > 0.005
    # print(flag.tolist())
    print("sum diff:", diff.sum().item())
    print("max diff:", diff.max().item())
    print("mean diff:", diff.mean().item())
    print(f"Triton time: {time_end - time_start}")
    # print(similarity_score.min(), similarity_score.max())

    diff_hf = (logits_hf - logits).abs()
    print("sum diff_hf:", diff_hf.sum().item())
    print("max diff_hf:", diff_hf.max().item())
    print("mean diff_hf:", diff_hf.mean().item())


if __name__ == "__main__":
    test()
