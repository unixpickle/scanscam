import itertools
from typing import Sequence

import pytest
import torch

from scanscam.baseline import naive_linear_scan_backward, naive_linear_scan_forward
from scanscam.cuda import (
    blocked_linear_scan_backward,
    blocked_linear_scan_forward,
    coalesced_linear_scan_forward,
    simple_linear_scan_backward,
    simple_linear_scan_forward,
    transposed_linear_scan_backward,
    transposed_linear_scan_forward,
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
    "shape,permute",
    itertools.product(
        [(32, 5), (16, 2, 5), (32, 64), (1, 1024), (32, 1024), (32, 1025), (1, 1025)],
        [False, True],
    ),
)
def test_simple_linear_scan_backward(shape: Sequence[int], permute: bool):
    x = torch.randn(*shape, device="cuda")
    y = torch.randn(*shape, device="cuda")
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
    "shape,permute",
    itertools.product(
        [(32, 5), (16, 2, 5), (32, 64), (1, 1024), (32, 1024), (32, 1025), (1, 1025)],
        [False, True],
    ),
)
def test_blocked_linear_scan_backward(shape: Sequence[int], permute: bool):
    x = torch.randn(*shape, device="cuda")
    y = torch.randn(*shape, device="cuda")
    if permute:
        x = x.mT.contiguous().mT
    output = simple_linear_scan_forward(x, y)
    if permute:
        output = output.mT.contiguous().mT
    expected_gate_grad, expected_value_grad = naive_linear_scan_backward(x, y, output)
    actual_gate_grad, actual_value_grad = blocked_linear_scan_backward(x, y, output)
    assert torch.allclose(actual_gate_grad, expected_gate_grad, atol=1e-4)
    assert torch.allclose(
        actual_value_grad, expected_value_grad, atol=1e-4
    ), f"MAE = {(actual_value_grad-expected_value_grad).abs().max().item()}"


@pytest.mark.parametrize(
    "shape,channels_per_block",
    itertools.product(
        [(8, 4096, 512), (512, 4096, 8), (3, 107, 197), (3, 293, 283), (5, 6113, 107)],
        [1, 8, 32],
    ),
)
def test_transposed_linear_scan(shape: Sequence[int], channels_per_block: int):
    x = torch.randn(*shape, device="cuda")
    y = torch.randn(*shape, device="cuda")
    expected = naive_linear_scan_forward(x.mT, y.mT).mT
    actual = transposed_linear_scan_forward(x, y, channels_per_block)
    assert torch.allclose(actual, expected, atol=1e-4), f"{actual=} {expected=}"


@pytest.mark.parametrize(
    "shape,channels_per_block",
    itertools.product(
        [(8, 4096, 512), (512, 4096, 8), (3, 107, 197), (3, 293, 283), (5, 6113, 107)],
        [1, 8, 32],
    ),
)
def test_transposed_linear_scan_backward(shape: Sequence[int], channels_per_block: int):
    x = torch.randn(*shape, device="cuda")
    y = torch.randn(*shape, device="cuda")
    out = naive_linear_scan_forward(x.mT, y.mT).mT
    out_grad = torch.randn_like(out)
    expected_grads = [
        x.mT for x in naive_linear_scan_backward(x.mT, out.mT, out_grad.mT)
    ]
    actual_grads = transposed_linear_scan_backward(
        x, out.contiguous(), out_grad.contiguous(), channels_per_block
    )
    for i, (x, a) in enumerate(zip(expected_grads, actual_grads)):
        assert torch.allclose(a, x, atol=1e-4), f"{i=} actual={a} expected={x}"


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


@pytest.mark.parametrize(
    "shape",
    [(8, 256, 4096), (8, 256, 32), (256, 4096), (16384, 4096)],
)
def test_simple_linear_scan_backward_time(benchmark, shape):
    x = torch.randn(shape, device="cuda")
    y = torch.randn(shape, device="cuda")
    z = torch.randn(shape, device="cuda")
    torch.cuda.synchronize()

    def fn():
        simple_linear_scan_backward(x, y, z)
        torch.cuda.synchronize()

    benchmark(fn)


@pytest.mark.parametrize(
    "shape",
    [(8, 256, 4096), (8, 256, 32), (256, 4096), (16384, 4096)],
)
def test_blocked_linear_scan_backward_time(benchmark, shape):
    x = torch.randn(shape, device="cuda")
    y = torch.randn(shape, device="cuda")
    z = torch.randn(shape, device="cuda")
    torch.cuda.synchronize()

    def fn():
        blocked_linear_scan_backward(x, y, z)
        torch.cuda.synchronize()

    benchmark(fn)


@pytest.mark.parametrize(
    "shape,channels_per_block",
    itertools.product(
        [(8, 4096, 512)],
        [1, 2, 4, 8, 16, 32],
    ),
)
def test_transposed_linear_scan_time(
    benchmark, shape: Sequence[int], channels_per_block: int
):
    x = torch.randn(*shape, device="cuda")
    y = torch.randn(*shape, device="cuda")
    torch.cuda.synchronize()

    def fn():
        transposed_linear_scan_forward(x, y, channels_per_block)
        torch.cuda.synchronize()

    benchmark(fn)


@pytest.mark.parametrize(
    "shape,channels_per_block",
    itertools.product(
        [(8, 4096, 512)],
        [1, 2, 4, 8, 16, 32],
    ),
)
def test_transposed_linear_scan_backward_time(
    benchmark, shape: Sequence[int], channels_per_block: int
):
    x = torch.randn(*shape, device="cuda")
    y = torch.randn(*shape, device="cuda")
    z = torch.randn(*shape, device="cuda")
    torch.cuda.synchronize()

    def fn():
        transposed_linear_scan_backward(x, y, z, channels_per_block)
        torch.cuda.synchronize()

    benchmark(fn)


@pytest.mark.parametrize("naive", [False, True])
def test_transposed_linear_scan_fwd_bwd_time(benchmark, naive: bool):
    x = torch.randn(8, 4096, 512, device="cuda")
    y = torch.randn(8, 4096, 512, device="cuda")
    z = torch.randn(8, 4096, 512, device="cuda")
    channels_per_block = 32
    torch.cuda.synchronize()

    def fn():
        if naive:
            out = blocked_linear_scan_forward(x.mT, y.mT).mT
            g1, g2 = blocked_linear_scan_backward(x.mT, out.mT, z.mT)
        else:
            out = transposed_linear_scan_forward(x, y, channels_per_block)
            g1, g2 = transposed_linear_scan_backward(x, out, z, channels_per_block)
        (g1 + g2).sum().item()  # force usage

    benchmark(fn)
