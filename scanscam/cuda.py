from typing import Tuple

import torch

from scanscam.decorator import preprocess_backward_scan_args, preprocess_scan_args

try:
    import scanscam_ext

    has_cuda_ops = hasattr(scanscam_ext, "simple_linear_scan")
except ImportError:
    scanscam_ext = None
    has_cuda_ops = False


@preprocess_scan_args
def simple_linear_scan_forward(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    y = y.contiguous()
    out = torch.empty_like(x)
    scanscam_ext.simple_linear_scan(x, y, out)
    return out


@preprocess_scan_args
def coalesced_linear_scan_forward(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    y = y.contiguous()
    out = torch.empty_like(x)
    scanscam_ext.coalesced_linear_scan(x, y, out)
    return out


@preprocess_scan_args
def blocked_linear_scan_forward(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    y = y.contiguous()
    out = torch.empty_like(x)
    scanscam_ext.blocked_linear_scan(x, y, out)
    return out


@preprocess_backward_scan_args
def simple_linear_scan_backward(
    x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    x = x.contiguous()
    y = y.contiguous()
    z = z.contiguous()
    out_a = torch.empty_like(x)
    out_b = torch.empty_like(x)
    scanscam_ext.simple_linear_scan_backward(x, y, z, out_a, out_b)
    return out_a, out_b


@preprocess_backward_scan_args
def blocked_linear_scan_backward(
    x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    x = x.contiguous()
    y = y.contiguous()
    z = z.contiguous()
    out_a = torch.empty_like(x)
    out_b = torch.empty_like(x)
    scanscam_ext.blocked_linear_scan_backward(x, y, z, out_a, out_b)
    return out_a, out_b


def transposed_linear_scan_forward(
    x: torch.Tensor, y: torch.Tensor, channels_per_block: int
) -> torch.Tensor:
    assert x.shape == y.shape
    assert len(x.shape) == 3
    assert x.is_contiguous()
    assert y.is_contiguous()
    out = torch.empty_like(x)
    scanscam_ext.transposed_linear_scan(x, y, out, channels_per_block)
    return out


def transposed_linear_scan_backward(
    x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, channels_per_block: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.shape == y.shape
    assert x.shape == z.shape
    assert len(x.shape) == 3
    assert x.is_contiguous()
    assert y.is_contiguous()
    assert z.is_contiguous()
    out_a = torch.empty_like(x)
    out_b = torch.empty_like(x)
    scanscam_ext.transposed_linear_scan_backward(
        x, y, z, out_a, out_b, channels_per_block
    )
    return out_a, out_b
