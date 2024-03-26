from typing import Tuple

import torch

from scanscam.decorator import preprocess_backward_scan_args, preprocess_scan_args

try:
    import scanscam_ext

    has_cpu_ops = hasattr(scanscam_ext, "simple_linear_scan_cpu")
except ImportError:
    scanscam_ext = None
    has_cpu_ops = False


@preprocess_scan_args
def simple_linear_scan_forward(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    y = y.contiguous()
    out = torch.empty_like(x)
    scanscam_ext.simple_linear_scan_cpu(x, y, out)
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
    scanscam_ext.simple_linear_scan_backward_cpu(x, y, z, out_a, out_b)
    return out_a, out_b
