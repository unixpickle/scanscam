from typing import Tuple

import torch

from scanscam.decorator import preprocess_backward_scan_args, preprocess_scan_args


@preprocess_scan_args
def naive_linear_scan_forward(gate: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    acc = torch.zeros_like(gate[:, 0])
    results = []
    for g, v in zip(gate.unbind(1), value.unbind(1)):
        acc = acc * g + v
        results.append(acc)
    return torch.stack(results, dim=1)


@preprocess_backward_scan_args
def naive_linear_scan_backward(
    gate: torch.Tensor,
    output: torch.Tensor,
    out_grad: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    doutput = torch.zeros_like(gate[:, 0])

    gate_grads = []
    value_grads = []

    for i in range(out_grad.shape[1] - 1, -1, -1):
        prev_output = output[:, i - 1] if i > 0 else torch.zeros_like(doutput)
        doutput = doutput + out_grad[:, i]

        this_gate = gate[:, i]

        value_grads.append(doutput)
        gate_grads.append(prev_output * doutput)

        doutput = doutput * this_gate

    return torch.stack(gate_grads[::-1], dim=1), torch.stack(value_grads[::-1], dim=1)
