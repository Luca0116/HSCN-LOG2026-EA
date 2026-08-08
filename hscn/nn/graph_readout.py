"""Graph-level readout layers for batched HSCN embeddings."""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
from torch import nn
from torch_geometric.utils import scatter

READOUT_MODES = frozenset({"mean", "sum", "layerwise_hybrid"})


def layerwise_mean_max_pool(states: Sequence[torch.Tensor]) -> torch.Tensor:
    """Per-layer mean and max over nodes, concatenated across layers."""
    parts: List[torch.Tensor] = []
    for h in states:
        parts.append(h.mean(dim=0))
        parts.append(h.max(dim=0).values)
    return torch.cat(parts, dim=0)


def layerwise_readout_dim(
    num_layers: int,
    hidden_dim: int,
    in_dim: int,
    feature_mode: str,
    layer_hidden_dims: Optional[Sequence[int]] = None,
) -> int:
    """Output dim for H^(0)..H^(L) with mean||max per layer."""
    if layer_hidden_dims is not None:
        if len(layer_hidden_dims) != num_layers:
            raise ValueError(
                f"layer_hidden_dims length ({len(layer_hidden_dims)}) must equal "
                f"num_layers ({num_layers})"
            )
        conv_dims = [int(d) for d in layer_hidden_dims]
        h0_dim = conv_dims[0] if feature_mode == "mlp" else in_dim
        layer_dims = [h0_dim] + conv_dims
    else:
        h0_dim = hidden_dim if feature_mode == "mlp" else in_dim
        layer_dims = [h0_dim] + [hidden_dim] * num_layers
    return sum(2 * d for d in layer_dims)


class GraphReadout(nn.Module):
    """Pool node features into graph vectors (mean or sum)."""

    def __init__(
        self,
        hidden_dim: int,
        mode: str = "mean",
        dropout: float = 0.0,
        **_kwargs,
    ) -> None:
        super().__init__()
        del _kwargs

        if mode not in {"mean", "sum"}:
            raise ValueError(
                f"GraphReadout mode must be 'mean' or 'sum', got {mode!r}"
            )

        self.hidden_dim = hidden_dim
        self.mode = mode
        self.out_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        del z  # topology positions are unused for mean/sum pooling
        if h.size(0) != batch.size(0):
            raise ValueError("h and batch must have the same number of nodes.")

        batch = batch.to(device=h.device)
        dim_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        reduce = "sum" if self.mode == "sum" else "mean"
        out = scatter(h, batch, dim=0, dim_size=dim_size, reduce=reduce)
        return self.dropout(out)
