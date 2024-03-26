import itertools
from typing import Sequence

import pytest
import torch

from scanscam.baseline import naive_linear_scan_backward, naive_linear_scan_forward
from scanscam.cpu import simple_linear_scan_backward, simple_linear_scan_forward


@pytest.mark.parametrize(
    "shape,permute",
    itertools.product(
        [(32, 5), (16, 2, 5), (32, 64), (1, 1024), (32, 1024), (32, 1025), (1, 1025)],
        [False, True],
    ),
)
def test_simple_linear_scan_forward(shape: Sequence[int], permute: bool):
    x = torch.randn(*shape)
    y = torch.randn(*shape)
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
def test_simple_linear_scan_backward(shape: Sequence[int], permute: bool):
    x = torch.randn(*shape)
    y = torch.randn(*shape)
    if permute:
        x = x.mT.contiguous().mT
    output = simple_linear_scan_forward(x, y)
    if permute:
        output = output.mT.contiguous().mT
    expected_gate_grad, expected_value_grad = naive_linear_scan_backward(x, y, output)
    actual_gate_grad, actual_value_grad = simple_linear_scan_backward(x, y, output)
    assert torch.allclose(actual_gate_grad, expected_gate_grad, atol=1e-4)
    assert torch.allclose(
        actual_value_grad, expected_value_grad, atol=1e-4
    ), f"MAE = {(actual_value_grad-expected_value_grad).abs().max().item()}"


@pytest.mark.parametrize(
    "shape",
    [(8, 256, 4096), (8, 256, 32), (256, 4096), (16384, 4096)],
)
def test_simple_linear_scan_time(benchmark, shape):
    x = torch.randn(shape)
    y = torch.randn(shape)

    def fn():
        simple_linear_scan_forward(x, y)

    benchmark(fn)


@pytest.mark.parametrize(
    "shape",
    [(8, 256, 4096), (8, 256, 32), (256, 4096), (16384, 4096)],
)
def test_simple_linear_scan_backward_time(benchmark, shape):
    x = torch.randn(shape)
    y = torch.randn(shape)
    z = torch.randn(shape)

    def fn():
        simple_linear_scan_backward(x, y, z)

    benchmark(fn)
