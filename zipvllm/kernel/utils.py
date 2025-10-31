import torch


def get_padded_headsize(size):
    # Get closest power of 2 over or equal to 32.
    padded_d_model = 1 << (size - 1).bit_length()
    # Smallest head_dim supported is 16. If smaller, the tile in the
    # kernel is padded - there is no padding in memory for any dims.
    padded_d_model = max(padded_d_model, 16)
    return padded_d_model


def _strides(x: torch.Tensor, *stride_names: str):
    if x is None:
        return {f"stride_{s}": 0 for i, s in enumerate(stride_names)}

    assert x.ndim == len(stride_names)
    return {f"stride_{s}": x.stride(i) for i, s in enumerate(stride_names)}


@torch.compile
def topk_mask(input: torch.Tensor, k: int, dim: int = -1):
    """

    calculate if the last dimension of input is top-k, return a bool mask with the same shape
    top-k is True, others are False
    """
    topk_indices = torch.topk(input, k, dim=dim)[1]
    mask = torch.zeros_like(input, dtype=torch.bool)
    mask.scatter_(dim, topk_indices, True)
    return mask


@torch.compile
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
