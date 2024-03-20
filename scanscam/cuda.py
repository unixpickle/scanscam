from typing import Tuple

import torch

from scanscam.decorator import preprocess_backward_scan_args, preprocess_scan_args

try:
    import scanscam_cuda

    has_cuda_ops = True
except ModuleNotFoundError:
    scanscam_cuda = None
    has_cuda_ops = False


@preprocess_scan_args
def simple_linear_scan_forward(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    y = y.contiguous()
    out = torch.empty_like(x)
    scanscam_cuda.simple_linear_scan(x, y, out)
    return out


@preprocess_scan_args
def coalesced_linear_scan_forward(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    y = y.contiguous()
    out = torch.empty_like(x)
    scanscam_cuda.coalesced_linear_scan(x, y, out)
    return out


@preprocess_scan_args
def blocked_linear_scan_forward(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    y = y.contiguous()
    out = torch.empty_like(x)
    scanscam_cuda.blocked_linear_scan(x, y, out)
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
    scanscam_cuda.simple_linear_scan_backward(x, y, z, out_a, out_b)
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
    scanscam_cuda.blocked_linear_scan_backward(x, y, z, out_a, out_b)
    return out_a, out_b
