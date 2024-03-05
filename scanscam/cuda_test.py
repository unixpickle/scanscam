import itertools
from typing import Sequence

import pytest
import torch

from scanscam.baseline import naive_linear_scan_forward
from scanscam.cuda import simple_linear_scan_forward


@pytest.mark.parametrize(
    "shape,permute",
    itertools.product(
        [(32, 5), (16, 2, 5), (32, 64), (1, 1024), (32, 1024), (32, 1025), (1, 1025)],
        [False, True],
    ),
)
def test_simple_linear_scan_forward(shape: Sequence[int], permute: bool):
    x = torch.randn(*shape, device="cuda")
    y = torch.randn(*shape, device="cuda")
    if permute:
        x = x.mT.contiguous().mT
    actual = simple_linear_scan_forward(x, y)
    expected = naive_linear_scan_forward(x, y)
    assert torch.allclose(actual, expected, atol=1e-4)


def test_simple_linear_scan_time(benchmark):
    x = torch.randn(8, 256, 4096, device="cuda")
    y = torch.randn(8, 256, 4096, device="cuda")

    def fn():
        simple_linear_scan_forward(x, y)

    benchmark(fn)
