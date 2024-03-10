import itertools
from typing import Sequence

import pytest
import torch

from scanscam.baseline import naive_linear_scan_forward
from scanscam.cuda import (
    blocked_linear_scan_forward,
    coalesced_linear_scan_forward,
    simple_linear_scan_forward,
)


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


@pytest.mark.parametrize(
    "shape,permute",
    itertools.product(
        [(32, 5), (16, 2, 5), (32, 64), (1, 1024), (32, 1024), (32, 1025), (1, 1025)],
        [False, True],
    ),
)
def test_coalesced_linear_scan_forward(shape: Sequence[int], permute: bool):
    x = torch.randn(*shape, device="cuda")
    y = torch.randn(*shape, device="cuda")
    if permute:
        x = x.mT.contiguous().mT
    actual = coalesced_linear_scan_forward(x, y)
    expected = naive_linear_scan_forward(x, y)
    assert torch.allclose(actual, expected, atol=1e-4)


@pytest.mark.parametrize(
    "shape",
    [
        (32, 5),
        (16, 2, 5),
        (32, 64),
        (32, 128),
        (32, 256),
        (32, 512),
        (32, 513),
        (1, 1024),
        (32, 1024),
        (32, 1025),
        (1, 1025),
        (32, 4095),
        (32, 4096),
        (32, 4097),
        (32, 4100),
        (32, 16384),
        (32, 16388),
    ],
)
def test_blocked_linear_scan_forward(shape: Sequence[int]):
    x = torch.randn(*shape, device="cuda")
    y = torch.randn(*shape, device="cuda")
    actual = blocked_linear_scan_forward(x, y)
    expected = naive_linear_scan_forward(x, y)
    assert torch.allclose(actual, expected, atol=1e-4)


@pytest.mark.parametrize(
    "shape",
    [(8, 256, 4096), (8, 256, 32), (256, 4096), (16384, 4096)],
)
def test_simple_linear_scan_time(benchmark, shape):
    x = torch.randn(shape, device="cuda")
    y = torch.randn(shape, device="cuda")
    torch.cuda.synchronize()

    def fn():
        simple_linear_scan_forward(x, y)
        torch.cuda.synchronize()

    benchmark(fn)


@pytest.mark.parametrize(
    "shape",
    [(8, 256, 4096), (8, 256, 32), (256, 4096), (16384, 4096)],
)
def test_coalesced_linear_scan_time(benchmark, shape):
    x = torch.randn(shape, device="cuda")
    y = torch.randn(shape, device="cuda")
    torch.cuda.synchronize()

    def fn():
        coalesced_linear_scan_forward(x, y)
        torch.cuda.synchronize()

    benchmark(fn)


@pytest.mark.parametrize(
    "shape",
    [(8, 256, 4096), (8, 256, 32), (256, 4096), (16384, 4096)],
)
def test_blocked_linear_scan_time(benchmark, shape):
    x = torch.randn(shape, device="cuda")
    y = torch.randn(shape, device="cuda")
    torch.cuda.synchronize()

    def fn():
        blocked_linear_scan_forward(x, y)
        torch.cuda.synchronize()

    benchmark(fn)
