"""Shared MLP building block."""

from __future__ import annotations

from torch import nn


class MLP(nn.Module):
    def __init__(
        self,
        dims,
        activation: str = "gelu",
        dropout: float = 0.0,
        final_activation: bool = False,
    ) -> None:
        super().__init__()

        if activation == "relu":
            act = nn.ReLU
        elif activation == "silu":
            act = nn.SiLU
        elif activation == "gelu":
            act = nn.GELU
        elif activation == "mish":
            act = nn.Mish
        else:
            raise ValueError(f"Unknown activation: {activation}")

        layers = []

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))

            is_last = i == len(dims) - 2
            if (not is_last) or final_activation:
                layers.append(act())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
