import torch
import torch.nn as nn

from scanscam.baseline import naive_linear_scan_backward, naive_linear_scan_forward


def test_naive_linear_scan_backward():
    gates = nn.Parameter(torch.randn(8, 16))
    values = nn.Parameter(torch.randn(8, 16))
    output_grad = torch.randn_like(values)
    output = naive_linear_scan_forward(gates, values)
    (output * output_grad).sum().backward()

    with torch.no_grad():
        actual_gate_grad, actual_value_grad = naive_linear_scan_backward(
            gates, output.detach(), output_grad
        )

    assert torch.allclose(
        gates.grad, actual_gate_grad
    ), f"{gates.grad=} {actual_gate_grad=}"
    assert torch.allclose(
        values.grad, actual_value_grad
    ), f"{values.grad=} {actual_value_grad=}"
