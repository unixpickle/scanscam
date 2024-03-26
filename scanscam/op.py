from typing import Any, Tuple

import torch

from . import cpu, cuda
from .baseline import naive_linear_scan_backward, naive_linear_scan_forward


def scan(gate: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """
    Apply a first-order scan operation.

    :param gate: [N x ... x T] tensor of gates
    :param value: [N x ... x T] tensor of values
    :return: an [N x ... x T] tensor of accumulated values.
    """
    return Scan.apply(gate, value)


class Scan(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, gate: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        if gate.device.type == "cuda" and cuda.has_cuda_ops:
            output = cuda.blocked_linear_scan_forward(gate, value)
        elif gate.device.type == "cpu" and cpu.has_cpu_ops:
            output = cpu.simple_linear_scan_forward(gate, value)
        else:
            output = naive_linear_scan_forward(gate, value)
        ctx.save_for_backward(gate, output)
        return output

    @staticmethod
    def backward(
        ctx: Any, output_grad: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        gate, output = ctx.saved_tensors
        if output_grad.device.type == "cuda" and cuda.has_cuda_ops:
            return cuda.blocked_linear_scan_backward(gate, output, output_grad)
        elif gate.device.type == "cpu" and cpu.has_cpu_ops:
            output = cpu.simple_linear_scan_backward(gate, output, output_grad)
        else:
            return naive_linear_scan_backward(gate, output, output_grad)
