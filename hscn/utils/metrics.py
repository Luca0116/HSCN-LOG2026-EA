from __future__ import annotations

import torch


@torch.no_grad()
def accuracy(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    if mask.sum().item() == 0:
        return 0.0
    pred = logits.argmax(dim=-1)
    return (pred[mask] == y[mask]).float().mean().item()
