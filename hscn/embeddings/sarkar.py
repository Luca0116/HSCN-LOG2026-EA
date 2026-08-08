from __future__ import annotations

import math
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from typing import List, Optional, Sequence, Tuple, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch_geometric.data import Data

# BFS-tree support uses the 2D disk (polar coordinates).
SARKAR_BALL_DIM = 2


def _build_undirected_adjacency(
    edge_index: torch.Tensor,
    num_nodes: int,
) -> List[List[int]]:
    """Build a Python adjacency list from PyG edge_index."""
    edge_index_cpu = edge_index.detach().cpu()
    row = edge_index_cpu[0].tolist()
    col = edge_index_cpu[1].tolist()

    adjacency_sets = [set() for _ in range(num_nodes)]
    for u, v in zip(row, col):
        u, v = int(u), int(v)
        if u == v:
            continue
        adjacency_sets[u].add(v)
        adjacency_sets[v].add(u)

    return [sorted(list(nei)) for nei in adjacency_sets]


def _bfs_forest(
    adjacency: List[List[int]],
    root: Optional[int] = None,
) -> Tuple[List[int], List[List[int]], List[int], List[int]]:
    """Build a BFS spanning forest over disconnected components."""
    num_nodes = len(adjacency)
    parent = [-2 for _ in range(num_nodes)]
    depth = [0 for _ in range(num_nodes)]
    children: List[List[int]] = [[] for _ in range(num_nodes)]
    roots: List[int] = []

    if num_nodes == 0:
        return parent, children, depth, roots

    if root is None:
        first_root = max(range(num_nodes), key=lambda u: len(adjacency[u]))
    else:
        first_root = int(root)

    candidate_roots = [first_root] + [u for u in range(num_nodes) if u != first_root]

    for start in candidate_roots:
        if parent[start] != -2:
            continue

        parent[start] = -1
        depth[start] = 0
        roots.append(start)

        queue = deque([start])
        while queue:
            u = queue.popleft()
            for v in adjacency[u]:
                if parent[v] != -2:
                    continue
                parent[v] = u
                depth[v] = depth[u] + 1
                children[u].append(v)
                queue.append(v)

    return parent, children, depth, roots


def _compute_subtree_sizes(
    children: List[List[int]],
    roots: List[int],
) -> List[int]:
    """Compute subtree sizes of the BFS forest."""
    num_nodes = len(children)
    subtree_size = [1 for _ in range(num_nodes)]

    order: List[int] = []
    stack = list(roots)
    while stack:
        u = stack.pop()
        order.append(u)
        stack.extend(children[u])

    for u in reversed(order):
        total = 1
        for v in children[u]:
            total += subtree_size[v]
        subtree_size[u] = total

    return subtree_size


def _assign_angles_by_subtree(
    children: List[List[int]],
    roots: List[int],
    subtree_size: List[int],
) -> List[float]:
    """Assign angular coordinates recursively by subtree mass."""
    num_nodes = len(children)
    theta = [0.0 for _ in range(num_nodes)]

    total_size = sum(subtree_size[r] for r in roots)
    if total_size <= 0:
        return theta

    current = 0.0
    component_sectors: List[Tuple[int, float, float]] = []
    for r in roots:
        width = 2.0 * math.pi * subtree_size[r] / total_size
        start = current
        end = current + width
        component_sectors.append((r, start, end))
        current = end

    stack: List[Tuple[int, float, float]] = []
    for r, start, end in component_sectors:
        theta[r] = 0.5 * (start + end)
        stack.append((r, start, end))

    while stack:
        u, start, end = stack.pop()
        child_list = children[u]
        if len(child_list) == 0:
            continue

        total_child_size = sum(subtree_size[v] for v in child_list)
        if total_child_size <= 0:
            continue

        current = start
        for v in child_list:
            width = (end - start) * subtree_size[v] / total_child_size
            child_start = current
            child_end = current + width
            theta[v] = 0.5 * (child_start + child_end)
            current = child_end
            stack.append((v, child_start, child_end))

    return theta


def _bfs_layout_angles_and_depths(
    edge_index: torch.Tensor,
    num_nodes: int,
    root: Optional[int],
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Shared BFS forest layout: per-node depth and subtree-mass angle."""
    adjacency = _build_undirected_adjacency(edge_index, num_nodes)
    _parent, children, depth, roots = _bfs_forest(adjacency, root=root)
    subtree_size = _compute_subtree_sizes(children, roots)
    theta = _assign_angles_by_subtree(children, roots, subtree_size)
    depth_tensor = torch.tensor(depth, dtype=dtype, device=device)
    theta_tensor = torch.tensor(theta, dtype=dtype, device=device)
    return depth_tensor, theta_tensor


def sarkar_embedding(
    edge_index: torch.Tensor,
    num_nodes: int,
    dim: int = 2,
    radius: float = 1.0,
    tau: float = 1.0,
    root: Optional[int] = None,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Deterministic BFS-tree hyperbolic embedding inspired by Sarkar's construction.

    Radial coordinate from BFS depth; angular sectors from subtree mass.
    This is a topology-induced Poincaré support, not the recursive isometric
    placement of the original Sarkar algorithm.
    """
    if dim < 2:
        raise ValueError("Hyperbolic BFS embedding needs dim >= 2.")
    if radius <= 0:
        raise ValueError("radius must be positive.")
    if tau <= 0:
        raise ValueError("tau must be positive.")
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive.")

    if device is None:
        device = edge_index.device

    depth_tensor, theta_tensor = _bfs_layout_angles_and_depths(
        edge_index, num_nodes, root, dtype, device
    )

    hyperbolic_distance = tau * depth_tensor
    poincare_norm = radius * torch.tanh(hyperbolic_distance / (2.0 * radius))

    z = torch.zeros(num_nodes, dim, dtype=dtype, device=device)
    z[:, 0] = poincare_norm * torch.cos(theta_tensor)
    z[:, 1] = poincare_norm * torch.sin(theta_tensor)

    norm = z.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    max_norm = radius * 0.99999
    z = torch.where(norm >= max_norm, z / norm * max_norm, z)

    return z


def euclidean_tree_embedding(
    edge_index: torch.Tensor,
    num_nodes: int,
    dim: int = 2,
    tau: float = 1.0,
    root: Optional[int] = None,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Flat analogue: same BFS angles, Euclidean radius = tau * depth."""
    if dim < 2:
        raise ValueError("Euclidean tree embedding needs dim >= 2.")
    if tau <= 0:
        raise ValueError("tau must be positive.")
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive.")

    if device is None:
        device = edge_index.device

    depth_tensor, theta_tensor = _bfs_layout_angles_and_depths(
        edge_index, num_nodes, root, dtype, device
    )
    radial = tau * depth_tensor

    z = torch.zeros(num_nodes, dim, dtype=dtype, device=device)
    z[:, 0] = radial * torch.cos(theta_tensor)
    z[:, 1] = radial * torch.sin(theta_tensor)
    return z


def _sarkar_worker(payload: Tuple[torch.Tensor, int, float, float, Optional[int]]) -> torch.Tensor:
    edge_index, num_nodes, tau, radius, root = payload
    return sarkar_embedding(
        edge_index=edge_index,
        num_nodes=num_nodes,
        dim=SARKAR_BALL_DIM,
        radius=radius,
        tau=tau,
        root=root,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )


def _euclidean_worker(payload: Tuple[torch.Tensor, int, float, Optional[int]]) -> torch.Tensor:
    edge_index, num_nodes, tau, root = payload
    return euclidean_tree_embedding(
        edge_index=edge_index,
        num_nodes=num_nodes,
        dim=SARKAR_BALL_DIM,
        tau=tau,
        root=root,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )


def precompute_sarkar_for_graphs(
    graphs: Sequence["Data"],
    tau: float = 0.5,
    radius: float = 1.0,
    root: Optional[int] = None,
    num_workers: int = 0,
    verbose: bool = False,
) -> None:
    """Attach fixed hyperbolic BFS-tree positions to each graph as ``data.z``."""
    if len(graphs) == 0:
        return

    payloads = [
        (graph.edge_index.cpu(), int(graph.num_nodes), float(tau), float(radius), root)
        for graph in graphs
    ]

    if num_workers and num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            positions = list(pool.map(_sarkar_worker, payloads))
    else:
        positions = [_sarkar_worker(payload) for payload in payloads]

    for graph, z in zip(graphs, positions):
        graph.z = z

    if verbose:
        print(
            f"precomputed hyperbolic BFS-tree z for {len(graphs)} graphs "
            f"(tau={tau}, radius={radius}, workers={max(num_workers, 1)})"
        )


def precompute_euclidean_for_graphs(
    graphs: Sequence["Data"],
    tau: float = 0.5,
    root: Optional[int] = None,
    num_workers: int = 0,
    verbose: bool = False,
) -> None:
    """Attach fixed Euclidean BFS-tree positions to each graph as ``data.z``."""
    if len(graphs) == 0:
        return

    payloads = [
        (graph.edge_index.cpu(), int(graph.num_nodes), float(tau), root)
        for graph in graphs
    ]

    if num_workers and num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            positions = list(pool.map(_euclidean_worker, payloads))
    else:
        positions = [_euclidean_worker(payload) for payload in payloads]

    for graph, z in zip(graphs, positions):
        graph.z = z

    if verbose:
        print(
            f"precomputed Euclidean BFS-tree z for {len(graphs)} graphs "
            f"(tau={tau}, workers={max(num_workers, 1)})"
        )
