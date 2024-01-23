#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Denoising Diffusion Probabilistic Models (DDPMs) and their variants.
"""


from functools import partial
from typing import Optional, Tuple

import torch
from tqdm import tqdm

from model.backbones import (
    MLPResNetBackboneModel,
    ResnetEncoderModel,
    UNetBackboneModel,
)
from model.time_encoding import SinusoidalTimeEncoder


class DiffusionModel(torch.nn.Module):
    def __init__(
        self,
        backbone: str,
        time_steps: int,
        beta_1: float,
        beta_T: float,
        bps_dim: int,
        choir_dim: int,
        temporal_dim: int,
        rescale_input: bool,
        embed_full_choir: bool,
        y_embed_dim: Optional[int] = None,
    ):
        super().__init__()
        y_dim = (bps_dim * choir_dim) if y_embed_dim is not None else None
        bps_grid_len = round(bps_dim ** (1 / 3))
        backbones = {
            "mlp_resnet": (
                partial(MLPResNetBackboneModel, hidden_dim=4096, bps_dim=bps_dim),
                torch.nn.Sequential(
                    torch.nn.Linear(y_dim, 2 * y_dim),
                    torch.nn.BatchNorm1d(2 * y_dim),
                    torch.nn.GELU(),
                    torch.nn.Linear(2 * y_dim, 2 * y_dim),
                    torch.nn.BatchNorm1d(2 * y_dim),
                    torch.nn.GELU(),
                    torch.nn.Linear(2 * y_dim, y_embed_dim),
                )
                if y_embed_dim is not None
                else None,
            ),
            "3d_unet": (
                partial(
                    UNetBackboneModel,
                    bps_grid_len=bps_grid_len,
                    normalization="group",
                    norm_groups=32,
                    pooling="avg",
                    use_spatial_transformer=True,
                ),
                partial(
                    ResnetEncoderModel,
                    choir_dim=choir_dim * (2 if embed_full_choir else 1),
                    normalization="group",
                    norm_groups=16,
                    pooling="avg",
                    pool_all_features="spatial",
                )
                if y_embed_dim is not None
                else None,
            ),
        }
        assert backbone in backbones, f"Unknown backbone {backbone}"
        self.backbone = backbones[backbone][0](
            time_encoder=SinusoidalTimeEncoder(time_steps, temporal_dim),
            choir_dim=choir_dim,
            temporal_dim=temporal_dim,
            context_channels=y_embed_dim,
        )
        if y_dim is not None or y_embed_dim is not None:
            assert y_dim is not None and y_embed_dim is not None
        self.embedder = (
            backbones[backbone][1](
                bps_grid_len=bps_grid_len,
                embed_channels=y_embed_dim,
            )
            if y_embed_dim is not None
            else None
        )
        self.time_steps = time_steps
        self.beta = torch.nn.Parameter(
            torch.linspace(beta_1, beta_T, time_steps), requires_grad=False
        )
        self.alpha = torch.nn.Parameter(
            torch.exp(
                torch.tril(torch.ones((time_steps, time_steps)))
                @ torch.log(1 - self.beta)
            ),
            requires_grad=False,
        )
        self.sigma = torch.nn.Parameter(torch.sqrt(self.beta), requires_grad=False)
        assert not torch.isnan(self.alpha).any(), "Alphas contains nan"
        assert not (self.alpha < 0).any(), "Alphas contain neg"
        self._input_shape = None
        self._rescale_input = rescale_input

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._input_shape is None:
            self._input_shape = x.shape[1:]
        # print(f"Input range: [{x.min()}, {x.max()}]")
        if self._rescale_input:
            # From [0, 1] to [-1, 1]:
            x = 2 * x - 1
        # print(f"Input range after stdization: [{x.min()}, {x.max()}]")
        # print(f"Input shape: {x.shape}")
        # print(f"Y shape: {y.shape if y is not None else None}")
        # ===== Training =========
        # 1. Sample timestep t with shape (B, 1)
        t = (
            torch.randint(0, self.time_steps, (x.shape[0], 1))
            .to(x.device)
            .requires_grad_(False)
        )
        # 2. Sample the noise with shape x.shape
        eps = torch.randn_like(x).to(x.device).requires_grad_(False)
        # 3. Diffuse the image
        batched_alpha = self.alpha[t]
        diffused_x = (
            torch.sqrt(batched_alpha)[..., None] * x
            + torch.sqrt(1 - batched_alpha)[..., None] * eps
        )
        # 4. Predict the noise sample
        y_embed = self.embedder(y) if y is not None else None
        # print(f"Y embed shape: {y_embed.shape if y_embed is not None else None}")
        eps_hat = self.backbone(diffused_x, t, y_embed, debug=True)
        return eps_hat, eps

    def generate(self, n: int, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        def make_t_batched(
            t: int, n: int, y_embed: Optional[torch.Tensor]
        ) -> torch.Tensor:
            if y_embed is None:
                t_batch = torch.tensor(t).view(1, 1).repeat(n, 1).to(device)
            else:
                t_batch = (
                    torch.tensor(t)
                    .view(1, 1, 1)
                    .repeat(y_embed.shape[0], n, 1)
                    .to(device)
                )
            return t_batch

        # ===== Inference =======
        assert self._input_shape is not None, "Must call forward first"
        with torch.no_grad():
            device = next(self.parameters()).device
            y_embed = self.embedder(y) if y is not None else None
            z_current = torch.randn(n, *self._input_shape).to(device)
            if y_embed is not None:
                z_current = z_current[None, ...].repeat(y_embed.shape[0], 1, 1, 1)
            _in_shape = z_current.shape
            pbar = tqdm(total=self.time_steps, desc="Generating")
            for t in range(self.time_steps - 1, 0, -1):  # Reversed from T to 1
                eps_hat = self.backbone(
                    z_current,
                    make_t_batched(t, n, y_embed),
                    y_embed,
                )
                z_prev_hat = (1 / (torch.sqrt(1 - self.beta[t]))) * z_current - (
                    self.beta[t]
                    / (torch.sqrt(1 - self.alpha[t]) * torch.sqrt(1 - self.beta[t]))
                ) * eps_hat
                eps = torch.randn_like(z_current)
                z_current = z_prev_hat + eps * self.sigma[t]
                pbar.update()
            # Now for z_0:
            eps_hat = self.backbone(z_current, make_t_batched(t, n, y_embed), y_embed)
            x_hat = (1 / (torch.sqrt(1 - self.beta[0]))) * z_current - (
                self.beta[0]
                / (torch.sqrt(1 - self.alpha[0]) * torch.sqrt(1 - self.beta[0]))
            ) * eps_hat
            pbar.update()
            pbar.close()
            output = x_hat.view(*_in_shape)
            if self._rescale_input:
                # Back to [0, 1]:
                output = (output + 1) / 2
                # print(f"Output range: [{output.min()}, {output.max()}]")
                output = torch.clamp(output, 0 + 1e-5, 1 - 1e-5)
                # print(
                # f"Output range after stdization: [{output.min()}, {output.max()}]"
                # )
            else:
                # Clamp to [-1, 1]:
                # print(f"Output range: [{output.min()}, {output.max()}]")
                output = torch.clamp(output, -1 + 1e-5, 1 - 1e-5)
                # print(
                # f"Output range after stdization: [{output.min()}, {output.max()}]"
                # )
            return output


class LatentDiffusionModel(torch.nn.Module):
    def __init__(
        self,
        autoencoder: torch.nn.Module,
        backbone: torch.nn.Module,
        time_steps: int,
        beta_1: float,
        beta_T: float,
    ):
        super().__init__()
        self.encoder = autoencoder.encoder
        self.encoder.requires_grad_(False)
        self.decoder = autoencoder.decoder
        self.decoder.requires_grad_(False)
        self.diffusion_model = DiffusionModel(backbone, time_steps, beta_1, beta_T)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        latent_map = self.encoder(x)
        return self.diffusion_model(latent_map)

    def generate(self, n: int) -> torch.Tensor:
        return self.decoder(self.diffusion_model.generate(n))
