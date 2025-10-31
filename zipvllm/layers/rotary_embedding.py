import torch
from nanovllm.layers.rotary_embedding import RotaryEmbedding


def remove_rope(y: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    y1, y2 = torch.chunk(y.to(torch.float32), 2, dim=-1)
    x1 = y1 * cos + y2 * sin
    x2 = y2 * cos - y1 * sin
    return torch.cat((x1, x2), dim=-1).to(y.dtype)


@torch.compile
def remove_forward(
    self,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
):
    cos_sin = self.cos_sin_cache[positions]
    cos, sin = cos_sin.chunk(2, dim=-1)
    query = remove_rope(query, cos, sin)
    key = remove_rope(key, cos, sin)
    return query, key


RotaryEmbedding.remove_forward = remove_forward
