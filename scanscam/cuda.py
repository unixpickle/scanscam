import torch

from scanscam.decorator import preprocess_scan_args

try:
    import scanscam_cuda
except ModuleNotFoundError:
    scanscam_cuda = None


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
