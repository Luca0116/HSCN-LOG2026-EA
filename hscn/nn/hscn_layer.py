"""Helgason–Fourier / Euclidean-Fourier spectral convolution layer (paper)."""

from __future__ import annotations

import torch
from torch import nn

from hscn.nn.helgason import euclidean_plane_wave, helgason_plane_wave
from hscn.nn.quadrature import (
    make_lambda_quadrature,
    make_sphere_quadrature,
    make_diffusion_scales,
)

GEOMETRY_MODES = frozenset({"hyperbolic", "euclidean"})
MULTIPLIER_MODES = frozenset({"full", "flat"})


class HSCNLayer(nn.Module):
    """
    Spectral convolution with multi-scale heat filters.

    Pipeline:
        plane-wave analysis → spectral multiplier → synthesis → re/im mix

    ``geometry_mode``:
        hyperbolic — Helgason waves on a Poincaré support; η = λ² + (1-σ)²/t²
        euclidean  — Fourier waves on a flat BFS-tree support; η = λ²

    ``multiplier_mode``:
        full — Σ_k α_ijk exp(-ρ_k η(λ))
        flat — Σ_k α_ijk  (frequency-independent; paper ablation)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ball_dim: int,
        radius: float = 1.0,
        sigma: float = 0.0,
        num_lambdas: int = 16,
        num_directions: int = 8,
        num_scales: int = 4,
        lambda_max: float = 8.0,
        min_scale: float = 0.1,
        max_scale: float = 10.0,
        scale_mode: str = "octave",
        learnable_scales: bool = False,
        dropout: float = 0.0,
        activation: str = "gelu",
        complex_pair_mode: str = "reim",
        norm_mode: str = "pre",
        geometry_mode: str = "hyperbolic",
        multiplier_mode: str = "full",
    ) -> None:
        super().__init__()

        if learnable_scales:
            raise ValueError(
                "Paper release uses fixed diffusion scales "
                "(learnable_scales must be False)."
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

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ball_dim = ball_dim
        self.radius = float(radius)
        self.sigma = float(sigma)
        self.num_lambdas = num_lambdas
        self.num_directions = num_directions
        self.num_scales = num_scales
        self.lambda_max = float(lambda_max)
        self.scale_mode = scale_mode
        self.complex_pair_mode = "reim"
        self.norm_mode = "pre"
        self.geometry_mode = geometry_mode
        self.multiplier_mode = multiplier_mode

        lambdas, lambda_weights = make_lambda_quadrature(
            num_lambdas=num_lambdas,
            lambda_max=lambda_max,
        )
        directions, direction_weights = make_sphere_quadrature(
            num_directions=num_directions,
            dim=ball_dim,
        )
        scales = make_diffusion_scales(
            num_scales=num_scales,
            min_scale=min_scale,
            max_scale=max_scale,
            mode=scale_mode,
        )

        self.register_buffer("lambdas", lambdas)
        self.register_buffer("lambda_weights", lambda_weights)
        self.register_buffer("directions", directions)
        self.register_buffer("direction_weights", direction_weights)
        self.register_buffer("fixed_scales", scales)

        self.alpha = nn.Parameter(
            torch.empty(in_channels, out_channels, num_scales)
        )
        nn.init.xavier_uniform_(self.alpha)

        self.residual = nn.Linear(in_channels, out_channels, bias=False)
        self.norm = nn.LayerNorm(in_channels)
        self.dropout = nn.Dropout(dropout)

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "mish":
            self.activation = nn.Mish()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self.pair_mix = nn.Linear(2 * out_channels, out_channels, bias=False)
        with torch.no_grad():
            self.pair_mix.weight.zero_()
            for j in range(out_channels):
                self.pair_mix.weight[j, j] = 1.0

    def diffusion_scales(self) -> torch.Tensor:
        return self.fixed_scales

    def _heat_eta(self, lambdas: torch.Tensor) -> torch.Tensor:
        """Radial spectral symbol η(λ)."""
        if self.geometry_mode == "euclidean":
            return lambdas.pow(2)
        return lambdas.pow(2) + ((1.0 - self.sigma) ** 2) / (self.radius ** 2)

    def spectral_response(self) -> torch.Tensor:
        """Filter g_hat[m, i, j] for ``multiplier_mode`` in {full, flat}."""
        lambdas = self.lambdas
        alpha = self.alpha

        if self.multiplier_mode == "flat":
            # Parameter-matched flat multiplier: M(λ) = Σ_k α_ijk
            a_sum = alpha.sum(dim=-1)  # [I, O]
            return a_sum.unsqueeze(0).expand(lambdas.size(0), -1, -1).contiguous()

        scales = self.diffusion_scales()
        eta = self._heat_eta(lambdas)
        basis = torch.exp(-eta[:, None] * scales[None, :])
        return torch.einsum("iok,mk->mio", alpha, basis)

    def _forward_analysis(
        self,
        h: torch.Tensor,
        omega: torch.Tensor,
        e_neg: torch.Tensor,
    ) -> torch.Tensor:
        h_complex = h.to(dtype=e_neg.dtype)
        omega_complex = omega.to(dtype=e_neg.dtype)
        return torch.einsum("n,ni,nmr->mri", omega_complex, h_complex, e_neg)

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if h.dim() != 2:
            raise ValueError(f"h should have shape [N, d_in], got {tuple(h.shape)}")
        if z.dim() != 2:
            raise ValueError(f"z should have shape [N, ball_dim], got {tuple(z.shape)}")
        if h.size(0) != z.size(0):
            raise ValueError("h and z must have the same number of nodes.")
        if h.size(1) != self.in_channels:
            raise ValueError(
                f"Expected h.size(1)={self.in_channels}, got {h.size(1)}"
            )
        if z.size(1) != self.ball_dim:
            raise ValueError(
                f"Expected z.size(1)={self.ball_dim}, got {z.size(1)}"
            )

        num_nodes = h.size(0)
        h_spectral = self.norm(h)
        omega = torch.full(
            (num_nodes,),
            1.0 / max(num_nodes, 1),
            dtype=h.dtype,
            device=h.device,
        )

        if self.geometry_mode == "euclidean":
            e_pos = euclidean_plane_wave(
                x=z,
                lambdas=self.lambdas.to(z.device),
                directions=self.directions.to(z.device),
                sign=1,
            )
        else:
            e_pos = helgason_plane_wave(
                z=z,
                lambdas=self.lambdas.to(z.device),
                directions=self.directions.to(z.device),
                radius=self.radius,
                sign=1,
            )
        e_neg = e_pos.conj()

        g_hat = self.spectral_response().to(device=h.device, dtype=e_neg.dtype)
        f_hat = self._forward_analysis(h_spectral, omega, e_neg)
        pair_hat = torch.einsum("mri,mij->mrij", f_hat, g_hat)

        inv_weight = (
            self.lambda_weights.to(h.device)[:, None]
            * self.direction_weights.to(h.device)[None, :]
        ).to(dtype=e_neg.dtype)

        pair_out_complex = torch.einsum(
            "mr,mrij,nmr->nij",
            inv_weight,
            pair_hat,
            e_pos,
        )
        re_sum = pair_out_complex.real.sum(dim=1)
        im_sum = pair_out_complex.imag.sum(dim=1)
        spectral_out = self.pair_mix(torch.cat([re_sum, im_sum], dim=-1))

        return self.residual(h) + self.dropout(self.activation(spectral_out))
