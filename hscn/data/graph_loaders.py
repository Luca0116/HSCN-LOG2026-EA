"""Data loaders for TU graph classification."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree

from hscn.embeddings.sarkar import precompute_sarkar_for_graphs

# Social / collaboration TU graphs with no native node attributes.
# PathNN / Errica et al.: use one-hot node degrees instead of discrete labels.
DEGREE_FEATURE_DATASETS = frozenset(
    {
        "IMDB-BINARY",
        "IMDB-MULTI",
        "COLLAB",
        "REDDIT-BINARY",
        "REDDIT-MULTI-5K",
        "REDDIT-MULTI-12K",
    }
)


def stratified_graph_split(
    labels: torch.Tensor,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    train_size: Optional[int] = None,
    test_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return boolean train/val/test masks over graphs."""
    if train_size is not None or test_size is not None:
        return fixed_stratified_graph_split(
            labels,
            train_size=train_size,
            test_size=test_size,
            val_size=0 if val_ratio == 0 else None,
            seed=seed,
        )

    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("Need 0 <= val, test and val+test < 1.")

    generator = torch.Generator().manual_seed(seed)
    num_graphs = labels.size(0)
    train_mask = torch.zeros(num_graphs, dtype=torch.bool)
    val_mask = torch.zeros(num_graphs, dtype=torch.bool)
    test_mask = torch.zeros(num_graphs, dtype=torch.bool)

    for label in labels.unique(sorted=True):
        idx = (labels == label).nonzero(as_tuple=True)[0]
        idx = idx[torch.randperm(idx.numel(), generator=generator)]
        n = idx.numel()
        n_test = max(1, int(round(n * test_ratio))) if n >= 3 else max(0, n - 2)
        n_val = max(1, int(round(n * val_ratio))) if n - n_test >= 2 else max(0, n - n_test - 1)
        n_train = n - n_val - n_test
        if n_train <= 0:
            n_train = max(1, n - n_val - n_test)
        test_mask[idx[:n_test]] = True
        val_mask[idx[n_test : n_test + n_val]] = True
        train_mask[idx[n_test + n_val :]] = True

    unassigned = ~(train_mask | val_mask | test_mask)
    if unassigned.any():
        train_mask[unassigned] = True

    return train_mask, val_mask, test_mask


def fixed_stratified_graph_split(
    labels: torch.Tensor,
    train_size: Optional[int],
    test_size: Optional[int],
    val_size: Optional[int] = None,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fixed-size stratified split (e.g. MUTAG 150/18)."""
    if train_size is None and test_size is None:
        raise ValueError("At least one of train_size or test_size must be set.")

    generator = torch.Generator().manual_seed(seed)
    num_graphs = labels.size(0)
    train_mask = torch.zeros(num_graphs, dtype=torch.bool)
    val_mask = torch.zeros(num_graphs, dtype=torch.bool)
    test_mask = torch.zeros(num_graphs, dtype=torch.bool)

    for label in labels.unique(sorted=True):
        idx = (labels == label).nonzero(as_tuple=True)[0]
        idx = idx[torch.randperm(idx.numel(), generator=generator)]
        n = idx.numel()
        n_test = min(test_size or 0, n)
        n_val = min(val_size or 0, max(0, n - n_test))
        n_train = min(train_size or max(0, n - n_test - n_val), max(0, n - n_test - n_val))
        test_mask[idx[:n_test]] = True
        val_mask[idx[n_test : n_test + n_val]] = True
        train_mask[idx[n_test + n_val : n_test + n_val + n_train]] = True

    return train_mask, val_mask, test_mask


def _load_tu_dataset(dataset: str, data_root: str):
    """Load TUDataset while staying compatible with PyTorch 2.6 torch.load."""
    import torch as _torch
    from torch_geometric.datasets import TUDataset

    orig_load = _torch.load

    def patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return orig_load(*args, **kwargs)

    _torch.load = patched_load
    try:
        return TUDataset(root=data_root, name=dataset)
    finally:
        _torch.load = orig_load


def _one_hot_degree_features(
    graphs: List[Data],
    max_degree: Optional[int] = None,
) -> int:
    """Replace / set ``graph.x`` with one-hot of node degree."""
    deg_list: List[torch.Tensor] = []
    for g in graphs:
        n = int(g.num_nodes)
        if g.edge_index is None or g.edge_index.numel() == 0:
            deg = torch.zeros(n, dtype=torch.long)
        else:
            deg = degree(g.edge_index[0], num_nodes=n, dtype=torch.long)
        deg_list.append(deg)

    if max_degree is None:
        max_degree = int(max(int(d.max().item()) if d.numel() else 0 for d in deg_list))
    feat_dim = max_degree + 1
    for g, deg in zip(graphs, deg_list):
        deg_c = deg.clamp(max=max_degree)
        g.x = torch.nn.functional.one_hot(deg_c, num_classes=feat_dim).to(torch.float32)
    return feat_dim


def ensure_tu_node_features(
    dataset: str,
    graphs: List[Data],
    *,
    feature_policy: str = "auto",
    max_degree: Optional[int] = None,
) -> int:
    """Ensure each graph has float ``x``; return ``in_dim``."""
    name = dataset.upper()
    policy = feature_policy.lower()
    has_x = all(g.x is not None and g.x.numel() > 0 for g in graphs)

    if policy == "attr":
        if not has_x:
            raise ValueError(f"{dataset}: feature_policy=attr but graphs have no x")
        for g in graphs:
            g.x = g.x.float()
        return int(graphs[0].x.size(-1))

    if policy == "degree":
        return _one_hot_degree_features(graphs, max_degree=max_degree)

    if policy == "attr_degree":
        attrs = [
            g.x.float().clone() if (g.x is not None and g.x.numel() > 0) else None
            for g in graphs
        ]
        deg_dim = _one_hot_degree_features(graphs, max_degree=max_degree)
        out_dim = deg_dim
        for g, attr in zip(graphs, attrs):
            if attr is None:
                continue
            g.x = torch.cat([attr, g.x], dim=-1)
            out_dim = int(g.x.size(-1))
        return out_dim

    if policy == "auto":
        if (not has_x) or (name in DEGREE_FEATURE_DATASETS):
            return _one_hot_degree_features(graphs, max_degree=max_degree)
        for g in graphs:
            g.x = g.x.float()
        return int(graphs[0].x.size(-1))

    raise ValueError(f"Unknown feature_policy={feature_policy!r}")


def load_tu_graph_dataset(
    dataset: str,
    data_root: str,
    seed: int = 42,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    train_size: Optional[int] = None,
    test_size: Optional[int] = None,
    sarkar_tau: float = 0.5,
    sarkar_root: Optional[int] = None,
    sarkar_radius: float = 1.0,
    precompute_sarkar: bool = True,
    sarkar_num_workers: int = 0,
    feature_policy: str = "auto",
) -> Tuple[List[Data], int, int, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load a TUDataset benchmark and return stratified train/val/test masks."""
    ds = _load_tu_dataset(dataset, data_root)
    graphs = [ds[i] for i in range(len(ds))]
    labels = torch.tensor([int(g.y.item()) for g in graphs], dtype=torch.long)
    num_classes = int(labels.max().item()) + 1
    in_dim = ensure_tu_node_features(dataset, graphs, feature_policy=feature_policy)

    train_mask, val_mask, test_mask = stratified_graph_split(
        labels,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        train_size=train_size,
        test_size=test_size,
    )
    if precompute_sarkar:
        precompute_sarkar_for_graphs(
            graphs,
            tau=sarkar_tau,
            radius=sarkar_radius,
            root=sarkar_root,
            num_workers=sarkar_num_workers,
            verbose=True,
        )
    return graphs, num_classes, in_dim, train_mask, val_mask, test_mask


def make_graph_loaders(
    graphs: List[Data],
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    batch_size: int = 32,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build PyG DataLoaders for train/val/test graph subsets."""
    train_data = [graphs[i] for i in train_mask.nonzero(as_tuple=True)[0].tolist()]
    val_data = [graphs[i] for i in val_mask.nonzero(as_tuple=True)[0].tolist()]
    test_data = [graphs[i] for i in test_mask.nonzero(as_tuple=True)[0].tolist()]
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


def stratified_kfold_masks(
    labels: torch.Tensor,
    n_folds: int = 10,
    val_ratio_within_trainval: float = 0.1,
    seed: int = 42,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Stratified k-fold: per fold ~ (1-1/k - val_ratio) train, val_ratio val, 1/k test."""
    generator = torch.Generator().manual_seed(seed)
    num_graphs = labels.size(0)
    class_to_indices: dict = {}
    for idx in range(num_graphs):
        y = int(labels[idx].item())
        class_to_indices.setdefault(y, []).append(idx)

    fold_test_indices: List[List[int]] = [[] for _ in range(n_folds)]
    for _label, indices in class_to_indices.items():
        idx = torch.tensor(indices, dtype=torch.long)
        idx = idx[torch.randperm(idx.numel(), generator=generator)]
        splits = torch.chunk(idx, n_folds)
        for fold_id, chunk in enumerate(splits):
            fold_test_indices[fold_id].extend(chunk.tolist())

    masks: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for fold_id in range(n_folds):
        test_mask = torch.zeros(num_graphs, dtype=torch.bool)
        test_mask[fold_test_indices[fold_id]] = True
        trainval_mask = ~test_mask

        val_mask = torch.zeros(num_graphs, dtype=torch.bool)
        train_mask = torch.zeros(num_graphs, dtype=torch.bool)
        trainval_idx = trainval_mask.nonzero(as_tuple=True)[0]
        trainval_labels = labels[trainval_idx]

        for label in trainval_labels.unique(sorted=True):
            idx = trainval_idx[(trainval_labels == label)]
            idx = idx[torch.randperm(idx.numel(), generator=generator)]
            n = idx.numel()
            n_val = max(1, int(round(n * val_ratio_within_trainval))) if n >= 2 else 0
            val_mask[idx[:n_val]] = True
            train_mask[idx[n_val:]] = True

        unassigned = trainval_mask & ~(train_mask | val_mask)
        if unassigned.any():
            train_mask[unassigned] = True

        masks.append((train_mask, val_mask, test_mask))
    return masks


def load_tu_graphs(
    dataset: str,
    data_root: str,
    sarkar_tau: float = 0.5,
    sarkar_root: Optional[int] = None,
    sarkar_radius: float = 1.0,
    precompute_sarkar: bool = True,
    sarkar_num_workers: int = 0,
    feature_policy: str = "auto",
):
    """Load all graphs and metadata from a TU dataset with Sarkar positions."""
    ds = _load_tu_dataset(dataset, data_root)
    graphs = [ds[i] for i in range(len(ds))]
    labels = torch.tensor([int(g.y.item()) for g in graphs], dtype=torch.long)
    num_classes = int(labels.max().item()) + 1
    in_dim = ensure_tu_node_features(dataset, graphs, feature_policy=feature_policy)
    if precompute_sarkar:
        precompute_sarkar_for_graphs(
            graphs,
            tau=sarkar_tau,
            radius=sarkar_radius,
            root=sarkar_root,
            num_workers=sarkar_num_workers,
            verbose=True,
        )
    return graphs, labels, num_classes, in_dim
