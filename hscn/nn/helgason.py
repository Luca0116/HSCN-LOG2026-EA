"""Helgason plane waves on the Poincare ball."""

from __future__ import annotations

import torch


def helgason_plane_wave(
    z: torch.Tensor,
    lambdas: torch.Tensor,
    directions: torch.Tensor,
    radius: float = 1.0,
    sign: int = 1,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Compute e_{±λ,ξ;t}(z) for all nodes, frequencies, and directions.

    Args:
        z: Tensor of shape [N, n], already inside B_t^n.
        lambdas: Tensor of shape [M].
        directions: Tensor of shape [R, n], unit vectors on S^{n-1}.
        radius: Poincare ball radius t.
        sign: +1 for e_{λ,ξ;t}; -1 for e_{-λ,ξ;t}.

    Returns:
        Complex tensor with shape [N, M, R].

    Formula used:
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
    z_norm2 = (z_scaled * z_scaled).sum(dim=-1).clamp(max=1.0 - eps)  # [N]
    numerator = (1.0 - z_norm2).clamp_min(eps)  # [N]

    # denom[n, r] = ||xi_r - z_n / t||^2
    diff = directions.unsqueeze(0) - z_scaled.unsqueeze(1)
    denom = (diff * diff).sum(dim=-1).clamp_min(eps)  # [N, R]
    base = (numerator.unsqueeze(-1) / denom).clamp_min(eps)  # [N, R]
    log_base = torch.log(base).unsqueeze(1)  # [N, 1, R]

    real_part = torch.full_like(lambdas, (n - 1.0) / 2.0)
    imag_part = sign * lambdas * radius / 2.0
    exponent = torch.complex(real_part, imag_part)  # [M]

    return torch.exp(exponent.view(1, -1, 1) * log_base.to(exponent.dtype))
