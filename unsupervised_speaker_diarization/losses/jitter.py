import torch


def jitter_loss(inputs: torch.Tensor) -> torch.Tensor:
    assert inputs.ndim == 2
    k, T = inputs.size()
    loss = torch.tensor(0.0)
    for r in range(k):
        for t in range(1, T):
            loss = loss + torch.abs(inputs[r, t] - inputs[r, t - 1])

    return loss / (k * T)
