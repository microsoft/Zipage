from functools import lru_cache
import torch
from torch import nn
import math

_rope_cache = None


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


def _compute_default_rope_parameters(base, dim) -> tuple["torch.Tensor", float]:
    attention_factor = 1.0
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
    return inv_freq, attention_factor


def _compute_llama3_parameters(
    base, dim, **rope_kwargs
) -> tuple["torch.Tensor", float]:
    inv_freq, attention_factor = _compute_default_rope_parameters(base, dim)

    factor = rope_kwargs["factor"]  
    low_freq_factor = rope_kwargs["low_freq_factor"]
    high_freq_factor = rope_kwargs["high_freq_factor"]
    old_context_len = rope_kwargs["original_max_position_embeddings"]
    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = 2 * math.pi / inv_freq
    
    inv_freq_llama = torch.where(
        wavelen > low_freq_wavelen, inv_freq / factor, inv_freq
    )
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (
        high_freq_factor - low_freq_factor
    )
    smoothed_inv_freq = (
        1 - smooth_factor
    ) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

    return inv_freq_llama, attention_factor


ROPE_INIT_FUNCTIONS = {
    "default": _compute_default_rope_parameters,
    "llama3": _compute_llama3_parameters,
}


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        rope_scaling: dict | None = None,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        if rope_scaling:
            self.rope_type = rope_scaling.get(
                "rope_type", rope_scaling.get("type", "default")
            )
        else:
            self.rope_type = "default"
        
        assert self.rope_type in ROPE_INIT_FUNCTIONS
        inv_freq, self.attention_scaling = ROPE_INIT_FUNCTIONS[self.rope_type](
            base=base, dim=rotary_dim, **rope_scaling if rope_scaling else {}
        )

        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos() * self.attention_scaling
        sin = freqs.sin() * self.attention_scaling
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    global _rope_cache
    if _rope_cache is None:
        _rope_cache = RotaryEmbedding(
            head_size, rotary_dim, max_position, base, rope_scaling=rope_scaling
        )
    return _rope_cache
