from typing import Any, Tuple

import torch

from . import cpu, cuda
from .baseline import naive_linear_scan_backward, naive_linear_scan_forward


def is_transpose_contig(x: torch.Tensor) -> bool:
    return len(x.shape) == 3 and not x.is_contiguous() and x.mT.is_contiguous()


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
            if is_transpose_contig(gate) and is_transpose_contig(value):
                output = cuda.transposed_linear_scan_forward(gate.mT, value.mT, 32).mT
            else:
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
            if is_transpose_contig(gate) and is_transpose_contig(output):
                outs = cuda.blocked_linear_scan_backward(
                    gate.mT, output.mT, output_grad.mT.contiguous(), 32
                )
                return tuple(x.mT for x in outs)
            else:
                return cuda.blocked_linear_scan_backward(gate, output, output_grad)
        elif gate.device.type == "cpu" and cpu.has_cpu_ops:
            output = cpu.simple_linear_scan_backward(gate, output, output_grad)
        else:
            return naive_linear_scan_backward(gate, output, output_grad)
