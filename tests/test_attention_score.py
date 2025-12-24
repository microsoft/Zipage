import os

os.environ["TRITON_INTERPRET"] = "1"
import torch
import numpy as np
import sys
import time
from pathlib import Path
import math

torch.set_printoptions(precision=8, sci_mode=False)
np.set_printoptions(precision=8, suppress=True)

sys.path.insert(0, str(Path(__file__).parent.parent))

from zipage.kernel.attention_score import attention_score


@torch.no_grad()
def compute_attention_scores(
    query_states, key_states, pooling="max", attention_mask=None, return_logits=False
):
    """
    query_states: (bsz, q_heads, q_len, head_dim)
    key_states: (bsz, kv_heads, kv_cache_len, head_dim)
    attention_mask: attention mask (bsz, kv_cache_len)

    return: (bsz, kv_heads, q_len, kv_cache_len - q_len)
    """
    batch_size, q_heads, q_len, head_dim = query_states.shape
    kv_heads = key_states.shape[1]
    kv_cache_len = key_states.shape[2]
    query_group_size = q_heads // kv_heads

    # shape: [batch_size, kv_heads, query_group_size, q_len, head_dim]
    query_states = query_states.view(
        batch_size, kv_heads, query_group_size, q_len, head_dim
    )

    # shape: [batch_size, kv_heads, 1, kv_cache_len, head_dim]
    key_states = key_states.unsqueeze(2)

    # we first normalize the key_states for better numerical stability
    key_states = key_states / math.sqrt(head_dim)
    # shape: [batch_size, kv_heads, query_group_size, q_len, kv_cache_len]
    attn_weights = torch.matmul(query_states, key_states.transpose(3, 4))

    if attention_mask is not None and torch.any(attention_mask == 0):
        if attention_mask.dim() == 2:
            # build causal mask (bsz,1,kv_cache_len,kv_cache_len) from attention_mask (bsz,kv_cache_len)
            # shape: (kv_cache_len, kv_cache_len)
            causal_mask = torch.triu(
                torch.ones(
                    kv_cache_len,
                    kv_cache_len,
                    device=attn_weights.device,
                    dtype=torch.bool,
                ),
                diagonal=1,
            )
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(
                1
            )  # (1,1,kv_cache_len,kv_cache_len)
            # shape: (bsz,1,kv_cache_len,kv_cache_len)
            causal_mask = causal_mask.expand(batch_size, -1, -1, -1)
            mask = (
                (attention_mask == 0).unsqueeze(1).unsqueeze(1)
            )  # (bsz,1,1,kv_cache_len)
            causal_mask = causal_mask.masked_fill(mask, True)
            causal_mask = causal_mask.unsqueeze(2)[:, :, :, -q_len:, :]
            # shape: (bsz,kv_heads,query_group_size,q_len,kv_cache_len)
            attn_weights = attn_weights.masked_fill(causal_mask, -float("inf"))
        else:
            raise ValueError("attention_mask must be 2D")
    else:
        # shape: (q_len, q_len)
        # no left padding, query can see all key before it
        mask = torch.triu(
            torch.ones(q_len, q_len, device=attn_weights.device), diagonal=1
        ).bool()
        attn_weights[:, :, :, :, -q_len:].masked_fill_(mask, -float("inf"))

    if return_logits:
        logits = attn_weights.clone()

    attn_scores = torch.softmax(
        attn_weights - attn_weights.max(dim=-1, keepdim=True).values, dim=-1
    )
    # apply pooling over attention head
    if pooling == "mean":
        attn_scores = attn_scores.mean(dim=2)
    elif pooling == "max":
        attn_scores = attn_scores.max(dim=2).values
    else:
        raise ValueError("Pooling method not supported")

    if return_logits:
        return logits, attn_scores
    return attn_scores


def cal_attention_scores_hf(
    query_cache,
    key_cache,
    block_table,
    seq_idx,
):
    num_layers, num_kvcache_blocks, block_size, num_kv_heads, head_dim = key_cache.shape
    _, _, query_cache_len, num_attention_heads, _ = query_cache.shape
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
    query_states = torch.zeros(
        num_layers,
        batch_size,
        num_attention_heads,
        query_cache_len,
        head_dim,
        device=query_cache.device,
        dtype=query_cache.dtype,
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
            for h in range(num_attention_heads):
                query_states[l, z, h, :, :] = query_cache[l, seq_idx[z].item(), :, h, :]
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
    # (num_layers, batch_size, num_kv_heads, seq_len, head_dim)
    key_states = key_states.view(num_layers, batch_size, num_kv_heads, -1, head_dim)

    attention_scores = torch.zeros(
        num_layers,
        batch_size,
        num_kv_heads,
        query_cache_len,
        max_num_blocks_per_seq * block_size,
        device=key_cache.device,
        dtype=key_cache.dtype,
    )
    logits = torch.zeros(
        num_layers,
        batch_size,
        num_attention_heads,
        query_cache_len,
        max_num_blocks_per_seq * block_size,
        device=key_cache.device,
        dtype=key_cache.dtype,
    )

    for l in range(num_layers):
        # (batch_size, num_attention_heads, query_cache_len, seq_len)
        logits_, attention_scores_ = compute_attention_scores(
            query_states[l],
            key_states[l],
            attention_mask=attention_mask,
            pooling="max",
            return_logits=True,
        )
        logits[l] = logits_.reshape(
            batch_size,
            num_attention_heads,
            query_cache_len,
            max_num_blocks_per_seq * block_size,
        )
        attention_scores[l] = attention_scores_

    logits = logits.reshape(
        num_layers,
        batch_size,
        num_attention_heads,
        query_cache_len,
        max_num_blocks_per_seq,
        block_size,
    )
    attention_scores = attention_scores.reshape(
        num_layers,
        batch_size,
        num_kv_heads,
        query_cache_len,
        max_num_blocks_per_seq,
        block_size,
    )

    mapped_logits = logits.clone()
    mapped_attention_scores = attention_scores.clone()

    for z in range(batch_size):
        if padd_block[z].item() == 0:
            continue
        pos = padd_block[z].item()
        for m in range(max_num_blocks_per_seq):
            if block_table[z, m] != -1:
                mapped_logits[:, z, :, :, m, :] = logits[:, z, :, :, pos, :]
                mapped_attention_scores[:, z, :, :, pos, :] = attention_scores[
                    :, z, :, :, pos, :
                ]
                pos += 1
            else:
                mapped_logits[:, z, :, :, m, :] = -float("inf")
                mapped_attention_scores[:, z, :, :, m, :] = 0
    mapped_logits = mapped_logits.view(
        num_layers, batch_size, num_attention_heads, query_cache_len, -1
    )
    mapped_attention_scores = mapped_attention_scores.view(
        num_layers, batch_size, num_kv_heads, query_cache_len, -1
    )
    mapped_attention_scores = mapped_attention_scores.mean(dim=-2)
    return mapped_logits, mapped_attention_scores


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
    num_kvcache_blocks = 2000
    block_size = 256
    num_kv_heads = 2
    head_dim = 128
    max_num_seqs = 10
    query_cache_len = 24
    num_attention_heads = 16
    layer = 16
    num_layers = 4
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
        key_cache[1 : 1 + num_layers],
        query_cache[1 : 1 + num_layers],
        seq_idx,
        block_table,
        return_logits=True,
    )
    score_ref, logits_ref = attention_score_ref(
        key_cache[1 : 1 + num_layers],
        query_cache[1 : 1 + num_layers],
        seq_idx,
        block_table,
    )
    logits_hf, scores_hf = cal_attention_scores_hf(
        query_cache[1 : 1 + num_layers],
        key_cache[1 : 1 + num_layers],
        block_table,
        seq_idx,
    )

    logits[logits == float("-inf")] = 0
    logits_ref[logits_ref == float("-inf")] = 0
    logits_hf[logits_hf == float("-inf")] = 0
    
    print('logits diff with ref:')
    diff = (logits - logits_ref).abs()
    print("sum diff with ref:", diff.sum())
    print("max diff with ref:", diff.max())
    print("mean diff with ref:", diff.mean())
    
    print('logits diff with hf:')
    diff_hf = (logits_hf - logits).abs()
    print("sum diff_hf:", diff_hf.sum())
    print("max diff_hf:", diff_hf.max())
    print("mean diff_hf:", diff_hf.mean())


if __name__ == "__main__":
    test()
