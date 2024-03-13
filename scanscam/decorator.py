from typing import Callable, Tuple

import torch

ScanFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
ReverseScanFn = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor],
]


def preprocess_scan_args(fn: ScanFn) -> ScanFn:
    """
    Decorator to flatten the batch dimensions of arguments passed to a scan
    implementation.
    """

    def new_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.shape == y.shape
        assert len(x.shape) >= 2
        new_x = x.reshape(-1, x.shape[-1])
        new_y = y.reshape(-1, y.shape[-1])
        return fn(new_x, new_y).reshape(x.shape)

    return new_fn


def preprocess_backward_scan_args(fn: ReverseScanFn) -> ReverseScanFn:
    """
    Decorator to flatten the batch dimensions of arguments passed to a scan
    backward pass implementation.
    """

    def new_fn(
        x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert x.shape == y.shape
        assert x.shape == z.shape
        assert len(x.shape) >= 2
        new_x = x.reshape(-1, x.shape[-1])
        new_y = y.reshape(-1, y.shape[-1])
        new_z = z.reshape(-1, z.shape[-1])
        out_a, out_b = fn(new_x, new_y, new_z)
        return out_a.reshape(x.shape), out_b.reshape(x.shape)

    return new_fn
