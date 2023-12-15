#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Denoising Diffusion Probabilistic Models (DDPMs) and their variants.
"""


from typing import Tuple

import torch
from tqdm import tqdm

from model.backbones import MLPResNetBackboneModel
from model.time_encoding import SinusoidalTimeEncoder


class DiffusionModel(torch.nn.Module):
    def __init__(
        self,
        time_steps: int,
        beta_1: float,
        beta_T: float,
        bps_dim: int,
        temporal_dim: int,
    ):
        super().__init__()
        self.backbone = MLPResNetBackboneModel(
            SinusoidalTimeEncoder(time_steps, temporal_dim), bps_dim, temporal_dim
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

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._input_shape is None:
            self._input_shape = x.shape[1:]
        # print(f"Input range: [{x.min()}, {x.max()}]")
        # From [0, 1] to [-1, 1]:
        x = 2 * x - 1
        # print(f"Input range after stdization: [{x.min()}, {x.max()}]")
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
        eps_hat = self.backbone(diffused_x, t)
        return eps_hat, eps

    def generate(self, n: int) -> torch.Tensor:
        # ===== Inference =======
        assert self._input_shape is not None, "Must call forward first"
        with torch.no_grad():
            device = next(self.parameters()).device
            z_current = torch.randn(n, *self._input_shape).to(device)
            pbar = tqdm(total=self.time_steps, desc="Generating...")
            for t in range(self.time_steps - 1, 0, -1):  # Reversed from T to 1
                eps_hat = self.backbone(
                    z_current, torch.tensor(t).view(1, 1).repeat(n, 1).to(device)
                )
                z_prev_hat = (1 / (torch.sqrt(1 - self.beta[t]))) * z_current - (
                    self.beta[t]
                    / (torch.sqrt(1 - self.alpha[t]) * torch.sqrt(1 - self.beta[t]))
                ) * eps_hat
                eps = torch.randn_like(z_current)
                z_current = z_prev_hat + eps * self.sigma[t]
                pbar.update()
            # Now for z_0:
            eps_hat = self.backbone(
                z_current, torch.tensor(0).view(1, 1).repeat(n, 1).to(device)
            )
            x_hat = (1 / (torch.sqrt(1 - self.beta[0]))) * z_current - (
                self.beta[0]
                / (torch.sqrt(1 - self.alpha[0]) * torch.sqrt(1 - self.beta[0]))
            ) * eps_hat
            pbar.update()
            pbar.close()
            return x_hat.view(n, *self._input_shape)


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
