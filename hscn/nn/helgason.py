"""Plane waves: Helgason (Poincaré) and Euclidean Fourier."""

from __future__ import annotations

import torch


def euclidean_plane_wave(
    x: torch.Tensor,
    lambdas: torch.Tensor,
    directions: torch.Tensor,
    sign: int = 1,
) -> torch.Tensor:
    """Euclidean Fourier plane waves e_{±λ,ξ}(x) = exp(± i λ ⟨ξ, x⟩).

    Flat-space degeneration used by ``geometry_mode=euclidean``.
    """
    if sign not in (-1, 1):
        raise ValueError("sign must be +1 or -1")
    if x.dim() != 2:
        raise ValueError("x must have shape [N, n]")

    dtype = x.dtype
    device = x.device
    lambdas = lambdas.to(device=device, dtype=dtype)
    directions = directions.to(device=device, dtype=dtype)

    phase = torch.matmul(x, directions.transpose(0, 1))  # [N, R]
    angle = sign * lambdas.view(1, -1, 1) * phase.unsqueeze(1)
    zero = torch.zeros_like(angle)
    return torch.exp(torch.complex(zero, angle))


def helgason_plane_wave(
    z: torch.Tensor,
    lambdas: torch.Tensor,
    directions: torch.Tensor,
    radius: float = 1.0,
    sign: int = 1,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Compute e_{±λ,ξ;t}(z) for all nodes, frequencies, and directions.

    Formula:
        e_{λ,ξ;t}(x) = ((1 - ||x||^2/t^2) / ||ξ - x/t||^2)^((n-1+iλt)/2)
    """
    if sign not in (-1, 1):
        raise ValueError("sign must be +1 or -1")
    if z.dim() != 2:
        raise ValueError("z must have shape [N, n]")

    n = z.size(-1)
    dtype = z.dtype
    device = z.device
    lambdas = lambdas.to(device=device, dtype=dtype)
    directions = directions.to(device=device, dtype=dtype)

    z_scaled = z / radius
    z_norm2 = (z_scaled * z_scaled).sum(dim=-1).clamp(max=1.0 - eps)
    numerator = (1.0 - z_norm2).clamp_min(eps)

    diff = directions.unsqueeze(0) - z_scaled.unsqueeze(1)
    denom = (diff * diff).sum(dim=-1).clamp_min(eps)
    base = (numerator.unsqueeze(-1) / denom).clamp_min(eps)
    log_base = torch.log(base).unsqueeze(1)

    real_part = torch.full_like(lambdas, (n - 1.0) / 2.0)
    imag_part = sign * lambdas * radius / 2.0
    exponent = torch.complex(real_part, imag_part)

    return torch.exp(exponent.view(1, -1, 1) * log_base.to(exponent.dtype))
