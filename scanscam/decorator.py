from typing import Callable

import torch

ScanFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


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
