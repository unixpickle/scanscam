import torch

from scanscam.decorator import preprocess_scan_args


@preprocess_scan_args
def naive_linear_scan_forward(gate: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    acc = torch.zeros_like(gate[:, 0])
    results = []
    for g, v in zip(gate.unbind(1), value.unbind(1)):
        acc = acc * g + v
        results.append(acc)
    return torch.stack(results, dim=1)
