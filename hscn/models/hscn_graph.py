"""Graph-level HSCN classifier (paper TU graph classification)."""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import torch
from torch import nn
from torch_geometric.utils import subgraph

from hscn.embeddings.sarkar import (
    SARKAR_BALL_DIM,
    euclidean_tree_embedding,
    sarkar_embedding,
)
from hscn.nn.mlp import MLP
from hscn.nn.graph_readout import (
    GraphReadout,
    READOUT_MODES,
    layerwise_mean_max_pool,
    layerwise_readout_dim,
)
from hscn.nn.hscn_layer import GEOMETRY_MODES, HSCNLayer, MULTIPLIER_MODES

CLASSIFIER_MODES = frozenset({"mlp"})


def parse_layer_hidden_dims(
    value: Optional[Union[str, Sequence[int]]],
    *,
    num_layers: int,
    hidden_dim: int,
) -> Optional[List[int]]:
    """Parse optional per-layer channel widths; None means uniform ``hidden_dim``."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"none", "uniform", "null"}:
            return None
        dims = [int(x.strip()) for x in text.split(",") if x.strip()]
    else:
        dims = [int(x) for x in value]
    if len(dims) != num_layers:
        raise ValueError(
            f"layer_hidden_dims length ({len(dims)}) must equal num_layers ({num_layers})"
        )
    if any(d <= 0 for d in dims):
        raise ValueError(f"layer_hidden_dims must be positive, got {dims}")
    if dims == [hidden_dim] * num_layers:
        return None
    return dims


def resolve_channel_pattern(
    pattern: Optional[str],
    *,
    num_layers: int,
    hidden_dim: int,
) -> Optional[List[int]]:
    """Build per-layer widths from a named pattern."""
    if pattern is None or pattern in {"uniform", "none", ""}:
        return None
    h = int(hidden_dim)
    L = int(num_layers)
    half = max(h // 2, 16)
    wide = max(h + h // 2, h + 16)

    def _align(d: int, base: int = 8) -> int:
        return max(base, int(round(d / base) * base))

    if pattern == "expand":
        dims = [
            _align(int(round(h + (wide - h) * (i / max(L - 1, 1)))))
            for i in range(L)
        ]
    elif pattern == "taper":
        dims = [
            _align(int(round(wide + (half - wide) * (i / max(L - 1, 1)))))
            for i in range(L)
        ]
    elif pattern == "bottle":
        dims = [_align(h)] * L
        if L >= 3:
            for i in range(1, L - 1):
                dims[i] = _align(half)
        elif L == 2:
            dims[0] = _align(h)
            dims[1] = _align(half)
    elif pattern == "wide_mid":
        dims = [_align(h)] * L
        if L >= 3:
            mid = L // 2
            dims[mid] = _align(wide)
            if L >= 4:
                dims[mid - 1] = _align(wide)
        elif L == 2:
            dims[1] = _align(wide)
    elif pattern == "wide_last":
        dims = [_align(h)] * (L - 1) + [_align(wide)]
    elif pattern == "wide_first":
        dims = [_align(wide)] + [_align(h)] * (L - 1)
    else:
        raise ValueError(f"Unknown channel_pattern: {pattern!r}")
    return dims


def encode_batched_positions(
    edge_index: torch.Tensor,
    batch: torch.Tensor,
    ball_dim: int,
    radius: float,
    tau: float,
    root: Optional[int],
    dtype: torch.dtype,
    device: torch.device,
    geometry_mode: str = "hyperbolic",
) -> torch.Tensor:
    """Compute per-graph BFS-tree support positions for a PyG batch."""
    if geometry_mode not in GEOMETRY_MODES:
        raise ValueError(
            f"geometry_mode must be one of {sorted(GEOMETRY_MODES)}, "
            f"got {geometry_mode!r}"
        )
    num_nodes = batch.size(0)
    z = torch.zeros(num_nodes, ball_dim, dtype=dtype, device=device)
    num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0

    for graph_id in range(num_graphs):
        node_mask = batch == graph_id
        node_idx = node_mask.nonzero(as_tuple=True)[0]
        if node_idx.numel() == 0:
            continue

        sub_edge_index, _ = subgraph(
            node_idx,
            edge_index,
            relabel_nodes=True,
            num_nodes=num_nodes,
        )
        if geometry_mode == "euclidean":
            z_sub = euclidean_tree_embedding(
                edge_index=sub_edge_index,
                num_nodes=node_idx.numel(),
                dim=SARKAR_BALL_DIM,
                tau=tau,
                root=root,
                dtype=dtype,
                device=device,
            )
        else:
            z_sub = sarkar_embedding(
                edge_index=sub_edge_index,
                num_nodes=node_idx.numel(),
                dim=SARKAR_BALL_DIM,
                radius=radius,
                tau=tau,
                root=root,
                dtype=dtype,
                device=device,
            )
        z[node_idx] = z_sub

    return z


class HSCNGraphClassifier(nn.Module):
    """
    Graph classification with hyperbolic Helgason–Fourier spectral convolution.

    Pipeline (per graph in batch):

        edge_index -> Sarkar z_p
        x          -> optional feature MLP -> HSCNLayer stack -> h_p
        optional post-conv FFN
        {h_p}      -> GraphReadout (mean/sum) or layerwise_hybrid
        graph_emb  -> MLP classifier -> graph logits
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        ball_dim: int = SARKAR_BALL_DIM,
        radius: float = 1.0,
        sigma: float = 0.0,
        num_lambdas: int = 16,
        num_directions: int = 8,
        num_scales: int = 4,
        lambda_max: float = 8.0,
        min_scale: float = 0.05,
        max_scale: float = 5.0,
        scale_mode: str = "logspace",
        learnable_scales: bool = False,
        dropout: float = 0.5,
        conv_dropout: Optional[float] = None,
        encoder_dropout: Optional[float] = None,
        classifier_dropout: Optional[float] = None,
        activation: str = "gelu",
        norm_mode: str = "pre",
        sarkar_tau: float = 0.5,
        sarkar_root: Optional[int] = None,
        feature_mode: str = "mlp",
        readout_mode: str = "mean",
        readout_dropout: Optional[float] = None,
        classifier_mode: str = "mlp",
        classifier_num_hidden_layers: int = 2,
        classifier_num_layers: Optional[int] = None,
        classifier_hidden_dim: Optional[int] = None,
        complex_pair_mode: str = "reim",
        use_post_conv_mlp: bool = True,
        post_conv_mlp_hidden_dim: Optional[int] = None,
        post_conv_dropout: Optional[float] = None,
        layer_hidden_dims: Optional[Union[str, Sequence[int]]] = None,
        channel_pattern: Optional[str] = None,
        geometry_mode: str = "hyperbolic",
        multiplier_mode: str = "full",
    ) -> None:
        super().__init__()

        if feature_mode not in {"mlp", "direct_conv"}:
            raise ValueError(
                f"feature_mode must be 'mlp' or 'direct_conv', got {feature_mode!r}"
            )
        if classifier_mode not in CLASSIFIER_MODES:
            raise ValueError(
                f"classifier_mode must be one of {sorted(CLASSIFIER_MODES)}, "
                f"got {classifier_mode!r}"
            )
        if readout_mode not in READOUT_MODES:
            raise ValueError(
                f"readout_mode must be one of {sorted(READOUT_MODES)}, "
                f"got {readout_mode!r}"
            )
        if complex_pair_mode != "reim":
            raise ValueError(
                f"Paper release requires complex_pair_mode='reim', "
                f"got {complex_pair_mode!r}"
            )
        if norm_mode != "pre":
            raise ValueError(
                f"Paper release requires norm_mode='pre', got {norm_mode!r}"
            )
        if learnable_scales:
            raise ValueError(
                "Paper release uses fixed diffusion scales "
                "(learnable_scales must be False)."
            )
        if geometry_mode not in GEOMETRY_MODES:
            raise ValueError(
                f"geometry_mode must be one of {sorted(GEOMETRY_MODES)}, "
                f"got {geometry_mode!r}"
            )
        if multiplier_mode not in MULTIPLIER_MODES:
            raise ValueError(
                f"multiplier_mode must be one of {sorted(MULTIPLIER_MODES)}, "
                f"got {multiplier_mode!r}"
            )
        if ball_dim != SARKAR_BALL_DIM:
            raise ValueError(
                f"BFS-tree support embedding is {SARKAR_BALL_DIM}D; ball_dim must be "
                f"{SARKAR_BALL_DIM}, got {ball_dim}."
            )

        if classifier_num_layers is not None:
            classifier_num_hidden_layers = int(classifier_num_layers)

        if layer_hidden_dims is not None and channel_pattern not in (
            None,
            "",
            "uniform",
            "none",
        ):
            channel_pattern = None
        resolved_dims = parse_layer_hidden_dims(
            layer_hidden_dims, num_layers=num_layers, hidden_dim=hidden_dim
        )
        if resolved_dims is None and channel_pattern not in (
            None,
            "",
            "uniform",
            "none",
        ):
            resolved_dims = resolve_channel_pattern(
                channel_pattern, num_layers=num_layers, hidden_dim=hidden_dim
            )
        conv_dims = resolved_dims if resolved_dims is not None else [hidden_dim] * num_layers
        last_dim = int(conv_dims[-1])
        first_dim = int(conv_dims[0])

        self.in_dim = in_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer_hidden_dims = (
            None if resolved_dims is None else [int(d) for d in resolved_dims]
        )
        self.ball_dim = ball_dim
        self.radius = radius
        self.sarkar_tau = sarkar_tau
        self.sarkar_root = sarkar_root
        self.feature_mode = feature_mode
        self.readout_mode = readout_mode
        self.norm_mode = "pre"
        self.classifier_mode = "mlp"
        self.classifier_num_hidden_layers = max(1, int(classifier_num_hidden_layers))
        # Legacy alias kept for YAML / summary compatibility.
        self.classifier_num_layers = self.classifier_num_hidden_layers
        self.classifier_hidden_dim = (
            last_dim if classifier_hidden_dim is None else int(classifier_hidden_dim)
        )
        self.complex_pair_mode = "reim"
        self.use_post_conv_mlp = use_post_conv_mlp
        self.sigma = float(sigma)
        self.geometry_mode = geometry_mode
        self.multiplier_mode = multiplier_mode

        pool_dropout = dropout if readout_dropout is None else readout_dropout
        conv_dropout = dropout if conv_dropout is None else conv_dropout
        encoder_dropout = dropout if encoder_dropout is None else encoder_dropout
        classifier_dropout = dropout if classifier_dropout is None else classifier_dropout

        if feature_mode == "mlp":
            self.feature_encoder = MLP(
                [in_dim, first_dim, first_dim],
                activation=activation,
                dropout=encoder_dropout,
                final_activation=True,
            )
            layer_in_channels = [first_dim] + list(conv_dims[:-1])
            layer_out_channels = list(conv_dims)
        else:
            self.feature_encoder = None
            layer_in_channels = [in_dim] + list(conv_dims[:-1])
            layer_out_channels = list(conv_dims)

        self.layers = nn.ModuleList(
            [
                HSCNLayer(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    ball_dim=ball_dim,
                    radius=radius,
                    sigma=sigma,
                    num_lambdas=num_lambdas,
                    num_directions=num_directions,
                    num_scales=num_scales,
                    lambda_max=lambda_max,
                    min_scale=min_scale,
                    max_scale=max_scale,
                    scale_mode=scale_mode,
                    learnable_scales=False,
                    dropout=conv_dropout,
                    activation=activation,
                    norm_mode="pre",
                    complex_pair_mode="reim",
                    geometry_mode=geometry_mode,
                    multiplier_mode=multiplier_mode,
                )
                for in_ch, out_ch in zip(layer_in_channels, layer_out_channels)
            ]
        )

        post_dropout = conv_dropout if post_conv_dropout is None else post_conv_dropout
        if use_post_conv_mlp:
            mlp_hidden = (
                last_dim * 2
                if post_conv_mlp_hidden_dim is None
                else int(post_conv_mlp_hidden_dim)
            )
            self.post_conv_norm = nn.LayerNorm(last_dim)
            self.post_conv_mlp = MLP(
                [last_dim, mlp_hidden, last_dim],
                activation=activation,
                dropout=post_dropout,
                final_activation=False,
            )
        else:
            self.post_conv_norm = None
            self.post_conv_mlp = None

        if readout_mode == "layerwise_hybrid":
            self.graph_readout = None
            readout_dim = layerwise_readout_dim(
                num_layers=num_layers,
                hidden_dim=hidden_dim,
                in_dim=in_dim,
                feature_mode=feature_mode,
                layer_hidden_dims=self.layer_hidden_dims,
            )
        else:
            if self.layer_hidden_dims is not None:
                raise ValueError(
                    "layer_hidden_dims currently requires readout_mode="
                    "'layerwise_hybrid'"
                )
            self.graph_readout = GraphReadout(
                hidden_dim=last_dim,
                mode=readout_mode,
                dropout=pool_dropout,
            )
            readout_dim = self.graph_readout.out_dim

        self.classifier = self._build_classifier_head(
            readout_dim=readout_dim,
            num_classes=num_classes,
            classifier_num_hidden_layers=self.classifier_num_hidden_layers,
            classifier_hidden_dim=self.classifier_hidden_dim,
            activation=activation,
            dropout=classifier_dropout,
        )

    @staticmethod
    def _build_classifier_head(
        readout_dim: int,
        num_classes: int,
        classifier_num_hidden_layers: int,
        classifier_hidden_dim: int,
        activation: str,
        dropout: float,
    ) -> nn.Module:
        n_hidden = max(1, classifier_num_hidden_layers)
        dims = [readout_dim] + [classifier_hidden_dim] * n_hidden + [num_classes]
        return nn.Sequential(
            nn.LayerNorm(readout_dim),
            MLP(
                dims,
                activation=activation,
                dropout=dropout,
                final_activation=False,
            ),
        )

    def _run_conv_on_graph(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            h = layer(h=h, z=z)
        return h

    def _apply_post_conv_mlp(self, h: torch.Tensor) -> torch.Tensor:
        if self.post_conv_mlp is None or self.post_conv_norm is None:
            return h
        return h + self.post_conv_mlp(self.post_conv_norm(h))

    def _collect_conv_states_on_graph(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
    ) -> list[torch.Tensor]:
        states = [h]
        for layer in self.layers:
            h = layer(h=h, z=z)
            states.append(h)
        return states

    def encode_nodes(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if batch is None:
            raise ValueError("batch vector is required for graph classification.")
        if z is None:
            z = encode_batched_positions(
                edge_index=edge_index,
                batch=batch,
                ball_dim=self.ball_dim,
                radius=self.radius,
                tau=self.sarkar_tau,
                root=self.sarkar_root,
                dtype=x.dtype,
                device=x.device,
                geometry_mode=self.geometry_mode,
            )
        num_graphs = int(batch.max().item()) + 1
        h_parts = []
        for graph_id in range(num_graphs):
            node_mask = batch == graph_id
            x_g = x[node_mask]
            z_g = z[node_mask]
            if self.feature_encoder is not None:
                x_g = self.feature_encoder(x_g)
            h_parts.append(
                self._apply_post_conv_mlp(self._run_conv_on_graph(x_g, z_g))
            )
        return torch.cat(h_parts, dim=0)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if batch is None:
            raise ValueError("batch vector is required for graph classification.")

        if z is None:
            z = encode_batched_positions(
                edge_index=edge_index,
                batch=batch,
                ball_dim=self.ball_dim,
                radius=self.radius,
                tau=self.sarkar_tau,
                root=self.sarkar_root,
                dtype=x.dtype,
                device=x.device,
                geometry_mode=self.geometry_mode,
            )

        num_graphs = int(batch.max().item()) + 1
        h_parts = []
        graph_embs = []
        use_layerwise = self.readout_mode == "layerwise_hybrid"
        for graph_id in range(num_graphs):
            node_mask = batch == graph_id
            x_g = x[node_mask]
            z_g = z[node_mask]
            if self.feature_encoder is not None:
                x_g = self.feature_encoder(x_g)
            if use_layerwise:
                states = self._collect_conv_states_on_graph(x_g, z_g)
                states[-1] = self._apply_post_conv_mlp(states[-1])
                graph_embs.append(layerwise_mean_max_pool(states))
            else:
                h_parts.append(
                    self._apply_post_conv_mlp(self._run_conv_on_graph(x_g, z_g))
                )

        if use_layerwise:
            graph_emb = torch.stack(graph_embs, dim=0)
        else:
            h = torch.cat(h_parts, dim=0)
            graph_emb = self.graph_readout(h, z, batch)
        return self.classifier(graph_emb)
