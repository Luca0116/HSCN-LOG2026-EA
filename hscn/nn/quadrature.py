from __future__ import annotations

import math

import torch


def make_masked_node_quadrature_weights(
    num_nodes: int,
    mask: torch.Tensor,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Build node quadrature weights that sum only over masked nodes.

    Masked-out nodes receive weight 0 and do not contribute to the forward
    Helgason quadrature. ``HSCNLayer`` renormalizes weights to sum to 1.

    Args:
        num_nodes: number of graph nodes N.
        mask: bool tensor [N]; True marks nodes included in quadrature.
    """
    if mask.numel() != num_nodes:
        raise ValueError(
            f"mask length must equal num_nodes={num_nodes}, got {mask.numel()}"
        )
    if device is None:
        device = mask.device
    mask = mask.to(device=device, dtype=torch.bool)
    if not bool(mask.any()):
        raise ValueError("quadrature mask must include at least one node.")
    weights = torch.zeros(num_nodes, dtype=dtype, device=device)
    weights[mask] = 1.0
    return weights


def make_lambda_quadrature(
    num_lambdas: int,
    lambda_max: float,
    dtype: torch.dtype = torch.float32,
):
    """
    Simple trapezoidal quadrature on [0, lambda_max].

    Returns:
        lambdas: [M]
        weights: [M]
    """
    if num_lambdas <= 0:
        raise ValueError("num_lambdas must be positive.")
    if lambda_max <= 0:
        raise ValueError("lambda_max must be positive.")

    if num_lambdas == 1:
        lambdas = torch.tensor([0.0], dtype=dtype)
        weights = torch.tensor([lambda_max], dtype=dtype)
        return lambdas, weights

    lambdas = torch.linspace(0.0, float(lambda_max), steps=num_lambdas, dtype=dtype)
    step = float(lambda_max) / float(num_lambdas - 1)
    weights = torch.full((num_lambdas,), step, dtype=dtype)
    weights[0] *= 0.5
    weights[-1] *= 0.5
    return lambdas, weights


def make_sphere_quadrature(
    num_directions: int,
    dim: int,
    dtype: torch.dtype = torch.float32,
    seed: int = 12345,
):
    """
    Approximate integration over S^{dim-1}.

    For dim=2:
        use evenly spaced angles on the unit circle.

    For dim>2:
        use fixed random normalized Gaussian directions.

    Returns:
        directions: [R, dim]
        weights: [R]
    """
    if num_directions <= 0:
        raise ValueError("num_directions must be positive.")
    if dim <= 0:
        raise ValueError("dim must be positive.")

    if dim == 1:
        directions = torch.ones(num_directions, 1, dtype=dtype)
    elif dim == 2:
        angles = torch.linspace(
            0.0,
            2.0 * math.pi,
            steps=num_directions + 1,
            dtype=dtype,
        )[:-1]
        directions = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
    else:
        generator = torch.Generator()
        generator.manual_seed(seed)
        directions = torch.randn(
            num_directions,
            dim,
            generator=generator,
            dtype=dtype,
        )
        directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    weights = torch.full((num_directions,), 1.0 / float(num_directions), dtype=dtype)
    return directions, weights


def make_diffusion_scales(
    num_scales: int,
    min_scale: float = 0.1,
    max_scale: float = 10.0,
    mode: str = "logspace",
    dtype: torch.dtype = torch.float32,
):
    """
    Build fixed diffusion scales rho_k for the radial wavelet basis.

    Modes:
        logspace:
            rho_k on a uniform grid in log10([min_scale, max_scale]).
        linear:
            rho_k uniformly spaced in [min_scale, max_scale].
        octave:
            rho_k = 2^e_k with e_k uniform in [log2(min_scale), log2(max_scale)].
            Dyadic / octave-style spacing (endpoints pinned to min/max).

    Returns:
        scales: [K]
    """
    if num_scales <= 0:
        raise ValueError("num_scales must be positive.")
    if min_scale <= 0 or max_scale <= 0:
        raise ValueError("min_scale and max_scale must be positive.")
    if min_scale > max_scale:
        raise ValueError("min_scale should be <= max_scale.")

    allowed = {"logspace", "linear", "octave"}
    if mode not in allowed:
        raise ValueError(f"scale_mode must be one of {sorted(allowed)}, got {mode!r}")

    if num_scales == 1:
        return torch.tensor([float(min_scale)], dtype=dtype)

    if mode == "linear":
        return torch.linspace(
            float(min_scale),
            float(max_scale),
            steps=num_scales,
            dtype=dtype,
        )

    if mode == "octave":
        log2_min = math.log2(float(min_scale))
        log2_max = math.log2(float(max_scale))
        exponents = torch.linspace(log2_min, log2_max, steps=num_scales, dtype=dtype)
        return torch.pow(2.0, exponents)

    log_min = math.log10(float(min_scale))
    log_max = math.log10(float(max_scale))
    return torch.logspace(log_min, log_max, steps=num_scales, dtype=dtype)
