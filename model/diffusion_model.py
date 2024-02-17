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
    ContactUNetBackboneModel,
    MLPResNetBackboneModel,
    MLPResNetEncoderModel,
    MultiScaleResnetEncoderModel,
    PointNet2EncoderModel,
    ResnetEncoderModel,
    TransformerEncoderModel,
    UNetBackboneModel,
)
from model.pos_encoding import SinusoidalPosEncoder
from utils.dataset import fetch_gaussian_params_from_CHOIR


class ContactsBPSDiffusionModel(torch.nn.Module):
    def __init__(
        self,
        backbone: str,
        time_steps: int,
        beta_1: float,
        beta_T: float,
        bps_dim: int,
        temporal_dim: int,
        use_backbone_self_attn: bool = False,
        use_encoder_self_attn: bool = False,
        object_in_encoder: bool = False,
        contacts_hidden_dim: int = 1024,
        contacts_skip_connections: bool = False,
        y_embed_dim: Optional[int] = None,
        context_channels: Optional[int] = None,
    ):
        super().__init__()
        y_dim = (bps_dim * 2) if y_embed_dim is not None else None
        bps_grid_len = round(bps_dim ** (1 / 3))
        # TODO: Move all this crap to config (god I'm lazy to do that)
        backbones = {
            "3d_unet": (
                partial(
                    ContactUNetBackboneModel,
                    context_channels=context_channels or y_embed_dim,
                    bps_grid_len=bps_grid_len,
                    normalization="group",
                    norm_groups=16,
                    pooling="avg",
                    input_dim=2 if object_in_encoder else 1,
                    output_dim=1,
                    contacts_hidden_dim=contacts_hidden_dim,
                    contacts_skip_connections=contacts_skip_connections,
                    contacts_dim=9,
                    use_self_attention=use_backbone_self_attn,
                ),
                partial(
                    ResnetEncoderModel,
                    choir_dim=2,
                    normalization="group",
                    norm_groups=16,
                    pooling="avg",
                    pool_all_features="spatial",
                    use_self_attention=use_encoder_self_attn,
                )
                if y_embed_dim is not None
                else None,
            ),
        }
        assert backbone in backbones, f"Unknown backbone {backbone}"
        self.object_in_encoder = object_in_encoder
        self.use_contacts = "contact" in backbone
        self.backbone = backbones[backbone][0](
            time_encoder=SinusoidalPosEncoder(time_steps, temporal_dim),
            # TODO: Refactor the entire handling of choir_dim for x and for y.... This is a complete mess
            temporal_dim=temporal_dim,
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
        self._input_udf_shape, self._input_contacts_shape = None, None
        self.x_udf_mean, self.x_udf_std = None, None
        self.x_contacts_mean, self.x_contacts_std = None, None
        self.y_udf_mean, self.y_udf_std = None, None

    def set_dataset_stats(
        self,
        dataset: torch.utils.data.Dataset,
    ) -> None:
        self.x_udf_mean, self.x_udf_std = dataset.gt_udf_mean, dataset.gt_udf_std
        self.x_contacts_mean, self.x_contacts_std = (
            dataset.contacts_mean,
            dataset.contacts_std,
        )
        self.y_udf_mean, self.y_udf_std = dataset.noisy_udf_mean, dataset.noisy_udf_std

    def _standardize(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        return (x - mean.to(x.device)) / std.to(x.device)

    def _destandardize(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        return x * std.to(x.device) + mean.to(x.device)

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # ============= Preprocessing =============
        assert self.x_udf_mean is not None, "Must call set_dataset_stats first"
        if self.object_in_encoder:
            x[..., :2] = self._standardize(x[..., :2], self.x_udf_mean, self.x_udf_std)
            if self.use_contacts:
                x[..., 2:] = self._standardize(
                    x[..., 2:], self.x_contacts_mean, self.x_contacts_std
                )
        else:
            x[..., 0] = self._standardize(
                x[..., 0], self.x_udf_mean[1], self.x_udf_std[1]
            )
            if self.use_contacts:
                x[..., 1:] = self._standardize(
                    x[..., 1:], self.x_contacts_mean, self.x_contacts_std
                )
        y = (
            self._standardize(y, self.y_udf_mean, self.y_udf_std)
            if y is not None
            else None
        )
        # ===== Training =========
        # 1. Sample timestep t with shape (B, 1)
        t = (
            torch.randint(0, self.time_steps, (x.shape[0], 1))
            .to(x.device)
            .requires_grad_(False)
        )
        # 2. Sample the noise with shape x.shape
        x_udf = x[..., 1 if self.object_in_encoder else 0].unsqueeze(-1)
        # TODO: Just give the anchor parameters to this forward method? (eeeh this makes sense like
        # this as well) (Refactor label computation)
        x_contacts = fetch_gaussian_params_from_CHOIR(
            x,
            self.backbone.anchor_indices,
            self.backbone.n_repeats,
            self.backbone.n_anchors,
            choir_includes_obj=self.object_in_encoder,
        )
        x_contacts = x_contacts[:, :, 0, :].reshape(-1, 32 * 9)

        if self._input_udf_shape is None:
            # Save for the generation.
            self._input_udf_shape = x_udf.shape[1:]
            self._input_contacts_shape = x_contacts.shape[1:]

        eps_udf, eps_contacts = (
            torch.randn_like(x_udf).to(x_udf.device).requires_grad_(False),
            torch.randn_like(x_contacts).to(x_contacts.device).requires_grad_(False),
        )
        # 3. Diffuse the image
        batched_alpha = self.alpha[t]
        diffused_x_udf, diffused_x_contacts = (
            torch.sqrt(batched_alpha)[..., None] * x_udf
            + torch.sqrt(1 - batched_alpha)[..., None] * eps_udf
        ), (
            torch.sqrt(batched_alpha) * x_contacts
            + torch.sqrt(1 - batched_alpha) * eps_contacts
        )
        # 4. Predict the noise sample
        y_embed = self.embedder(y) if y is not None else None
        # print(f"Y embed shape: {y_embed.shape if y_embed is not None else None}")
        eps_hat_udf, eps_hat_contacts = self.backbone(
            torch.cat((x[..., 0].unsqueeze(-1), diffused_x_udf), dim=-1)
            if self.object_in_encoder
            else diffused_x_udf,
            diffused_x_contacts,
            t,
            y_embed,
            debug=True,
        )
        return {
            "udf": (eps_hat_udf, eps_udf),
            "contacts": (eps_hat_contacts, eps_contacts),
        }

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

        # ==== Preprocessing ====
        y = (
            self._standardize(y, self.y_udf_mean, self.y_udf_std)
            if y is not None
            else None
        )
        # ===== Inference =======
        assert (
            self._input_udf_shape is not None or self._input_contacts_shape is not None
        ), "Must call forward first"
        if self.object_in_encoder:
            # TODO: This is a hack. In the future the object will always be in y I think
            object_udf = y[..., 0].unsqueeze(-1)
            assert (
                object_udf is not None
            ), "Must provide object_udf when object_in_encoder is True"
        with torch.no_grad():
            device = next(self.parameters()).device
            y_embed = self.embedder(y) if y is not None else None
            z_udf_current, z_contacts_current = (
                torch.randn(n, *self._input_udf_shape).to(device),
                torch.randn(n, *self._input_contacts_shape).to(device),
            )
            if y_embed is not None:
                z_udf_current = z_udf_current[None, ...].repeat(
                    y_embed.shape[0], 1, 1, 1
                )
                z_contacts_current = z_contacts_current[None, ...].repeat(
                    y_embed.shape[0], 1, 1, 1
                )
            _in_udf_shape, _in_contacts_shape = (
                z_udf_current.shape,
                z_contacts_current.shape,
            )
            pbar = tqdm(total=self.time_steps, desc="Generating")
            for t in range(5, 0, -1):  # Reversed from T to 1
                eps_hat_udf, eps_hat_contacts = self.backbone(
                    torch.cat((object_udf, z_udf_current), dim=-1)
                    if self.object_in_encoder
                    else z_udf_current,
                    z_contacts_current,
                    make_t_batched(t, n, y_embed),
                    y_embed,
                )
                z_udf_prev_hat = (
                    1 / (torch.sqrt(1 - self.beta[t]))
                ) * z_udf_current - (
                    self.beta[t]
                    / (torch.sqrt(1 - self.alpha[t]) * torch.sqrt(1 - self.beta[t]))
                ) * eps_hat_udf
                z_contacts_prev_hat = (
                    1 / (torch.sqrt(1 - self.beta[t]))
                ) * z_contacts_current - (
                    self.beta[t]
                    / (torch.sqrt(1 - self.alpha[t]) * torch.sqrt(1 - self.beta[t]))
                ) * eps_hat_contacts
                eps_udf, eps_contacts = torch.randn_like(
                    z_udf_current
                ), torch.randn_like(z_contacts_current)
                z_udf_current, z_contacts_current = (
                    z_udf_prev_hat + eps_udf * self.sigma[t],
                    z_contacts_prev_hat + eps_contacts * self.sigma[t],
                )
                pbar.update()
            # Now for z_0:
            eps_hat_udf, eps_hat_contacts = self.backbone(
                torch.cat((object_udf, z_udf_current), dim=-1)
                if self.object_in_encoder
                else z_udf_current,
                z_contacts_current,
                make_t_batched(t, n, y_embed),
                y_embed,
            )
            x_hat_udf = (1 / (torch.sqrt(1 - self.beta[0]))) * z_udf_current - (
                self.beta[0]
                / (torch.sqrt(1 - self.alpha[0]) * torch.sqrt(1 - self.beta[0]))
            ) * eps_hat_udf
            x_hat_contacts = (
                1 / (torch.sqrt(1 - self.beta[0]))
            ) * z_contacts_current - (
                self.beta[0]
                / (torch.sqrt(1 - self.alpha[0]) * torch.sqrt(1 - self.beta[0]))
            ) * eps_hat_contacts
            pbar.update()
            pbar.close()
            # ====== Postprocessing ======
            x_hat_udf = x_hat_udf.view(*_in_udf_shape)
            x_hat_contacts = x_hat_contacts.view(*_in_contacts_shape).view(
                *_in_contacts_shape[:-2], 32, 9
            )

            x_hat_udf = self._destandardize(
                x_hat_udf, self.x_udf_mean[1], self.x_udf_std[1]
            )
            x_hat_contacts = self._destandardize(
                x_hat_contacts, self.x_contacts_mean, self.x_contacts_std
            )
            x_hat_udf = torch.clamp(x_hat_udf, 1e-5, 1 - 1e-5)
            # Clamp the Gaussian means to have a maximum norm of 20mm:
            # TODO: I don't know how to do that yet
            # x_hat_contacts[..., :3] = ?
            # Threshold the Gaussian covariances to be *activated* above 1e-5
            # (or something else? Try and see!):
            # TODO: Remove this? It's better to do it in the contacts fitting loss with a
            # hyperparameter!
            # x_hat_contacts[..., 3:][x_hat_contacts[..., 3:] < 1e-5] = 0.0
            # ============================
            return x_hat_udf, x_hat_contacts


class BPSDiffusionModel(torch.nn.Module):
    def __init__(
        self,
        backbone: str,
        time_steps: int,
        beta_1: float,
        beta_T: float,
        bps_dim: int,
        choir_dim: int,  # TODO: Rename this because it's confusing since not reflecting the actual CHOIR but just the part used (object + hand or just hand)
        temporal_dim: int,
        rescale_input: bool,
        embed_full_choir: bool,
        use_backbone_self_attn: bool = False,
        use_encoder_self_attn: bool = False,
        y_embed_dim: Optional[int] = None,
        context_channels: Optional[int] = None,
    ):
        super().__init__()
        y_dim = (bps_dim * choir_dim) if y_embed_dim is not None else None
        bps_grid_len = round(bps_dim ** (1 / 3))
        # TODO: Move all this crap to config (god I'm lazy to do that)
        backbones = {
            "mlp_resnet": (
                partial(
                    MLPResNetBackboneModel,
                    hidden_dim=1024,
                    bps_dim=bps_dim,
                    context_channels=context_channels or y_embed_dim,
                ),
                partial(MLPResNetEncoderModel, hidden_dim=512, embed_dim=y_embed_dim)
                if y_embed_dim is not None
                else None,
            ),
            "3d_unet": (
                partial(
                    UNetBackboneModel,
                    context_channels=context_channels or y_embed_dim,
                    bps_grid_len=bps_grid_len,
                    normalization="group",
                    norm_groups=16,
                    pooling="avg",
                    use_self_attention=use_backbone_self_attn,
                ),
                partial(
                    ResnetEncoderModel,
                    choir_dim=choir_dim * 2,
                    normalization="group",
                    norm_groups=16,
                    pooling="avg",
                    pool_all_features="spatial",
                    use_self_attention=use_encoder_self_attn,
                )
                if y_embed_dim is not None
                else None,
            ),
            "3d_contact_unet": (
                partial(
                    ContactUNetBackboneModel,
                    context_channels=context_channels or y_embed_dim,
                    bps_grid_len=bps_grid_len,
                    normalization="group",
                    norm_groups=16,
                    pooling="avg",
                    use_self_attention=use_backbone_self_attn,
                    choir_dim=10,
                    no_decoding=True,
                ),
                partial(
                    ResnetEncoderModel,
                    choir_dim=choir_dim * 2,
                    normalization="group",
                    norm_groups=16,
                    pooling="avg",
                    pool_all_features="spatial",
                    use_self_attention=use_encoder_self_attn,
                )
                if y_embed_dim is not None
                else None,
            ),
            "3d_contact_unet_w_decoding": (
                partial(
                    ContactUNetBackboneModel,
                    context_channels=context_channels or y_embed_dim,
                    bps_grid_len=bps_grid_len,
                    normalization="group",
                    norm_groups=16,
                    pooling="avg",
                    use_self_attention=use_backbone_self_attn,
                    choir_dim=10,
                ),
                partial(
                    ResnetEncoderModel,
                    choir_dim=choir_dim * 2,
                    normalization="group",
                    norm_groups=16,
                    pooling="avg",
                    pool_all_features="spatial",
                    use_self_attention=use_encoder_self_attn,
                )
                if y_embed_dim is not None
                else None,
            ),
            "3d_unet_multiscale": (
                partial(
                    UNetBackboneModel,
                    bps_grid_len=bps_grid_len,
                    normalization="group",
                    norm_groups=16,
                    pooling="avg",
                    use_self_attention=use_backbone_self_attn,
                    context_channels=[4, 8, 16, 32, 64],
                ),
                partial(
                    MultiScaleResnetEncoderModel,
                    choir_dim=choir_dim * 2,
                    normalization="group",
                    norm_groups=16,
                    pooling="avg",
                    use_self_attention=use_encoder_self_attn,
                )
                if y_embed_dim is not None
                else None,
            ),
            "3d_unet_w_transformer": (
                partial(
                    UNetBackboneModel,
                    bps_grid_len=bps_grid_len,
                    context_channels=context_channels or y_embed_dim,
                    normalization="group",
                    norm_groups=16,
                    pooling="avg",
                    use_self_attention=use_backbone_self_attn,
                ),
                partial(
                    TransformerEncoderModel,
                    choir_dim=choir_dim * 2,
                )
                if y_embed_dim is not None
                else None,
            ),
            "3d_unet_w_transformer_spatial_patches": (
                partial(
                    UNetBackboneModel,
                    context_channels=context_channels or y_embed_dim,
                    bps_grid_len=bps_grid_len,
                    normalization="group",
                    norm_groups=16,
                    pooling="avg",
                    use_self_attention=use_backbone_self_attn,
                ),
                partial(
                    TransformerEncoderModel,
                    choir_dim=choir_dim * 2,
                    spatialize_patches=True,
                )
                if y_embed_dim is not None
                else None,
            ),
        }
        assert backbone in backbones, f"Unknown backbone {backbone}"
        self.use_contacts = "contact" in backbone
        self.backbone = backbones[backbone][0](
            time_encoder=SinusoidalPosEncoder(time_steps, temporal_dim),
            choir_dim=(
                choir_dim * (2 if embed_full_choir else 1)
                if "contact" not in backbone
                else 10
            ),  # TODO: Fix this horror!!!
            # TODO: Refactor the entire handling of choir_dim for x and for y.... This is a complete mess
            output_channels=1 if "contact" not in backbone else 10,
            temporal_dim=temporal_dim,
        )
        if y_dim is not None or y_embed_dim is not None:
            assert y_dim is not None and y_embed_dim is not None
        self.embed_full_choir = embed_full_choir
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

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._input_shape is None:
            self._input_shape = x.shape[1:]
        # ===== Training =========
        # 1. Sample timestep t with shape (B, 1)
        t = (
            torch.randint(0, self.time_steps, (x.shape[0], 1))
            .to(x.device)
            .requires_grad_(False)
        )
        # 2. Sample the noise with shape x.shape
        if self.embed_full_choir:
            obj_bps = x[..., 0][..., None]
            x = x[..., 1:][..., None]
        eps = torch.randn_like(x).to(x.device).requires_grad_(False)
        # print(f"eps.shape: {eps.shape}")
        # 3. Diffuse the image
        batched_alpha = self.alpha[t]
        diffused_x = (
            torch.sqrt(batched_alpha)[..., None] * x
            + torch.sqrt(1 - batched_alpha)[..., None] * eps
        )
        if self.embed_full_choir:
            # Fuse back the full choir: object_bps + diffused_hand_bps
            diffused_x = torch.cat([obj_bps, diffused_x], dim=-1)
        # print(f"diffused_x.shape: {diffused_x.shape}")
        # 4. Predict the noise sample
        y_embed = self.embedder(y) if y is not None else None
        # print(f"Y embed shape: {y_embed.shape if y_embed is not None else None}")
        eps_hat = self.backbone(diffused_x, t, y_embed, debug=True)
        # print(f"eps_hat.shape: {eps_hat.shape}")
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
        if self.embed_full_choir:
            assert y is not None, "Must provide y when embed_full_choir is True"
        with torch.no_grad():
            device = next(self.parameters()).device
            y_embed = self.embedder(y) if y is not None else None
            z_current = torch.randn(n, *self._input_shape).to(device)
            if y_embed is not None:
                z_current = z_current[None, ...].repeat(y_embed.shape[0], 1, 1, 1)
            if self.embed_full_choir:
                z_current[..., 0] = y[..., 0]
            _in_shape = z_current.shape
            pbar = tqdm(total=self.time_steps, desc="Generating")
            for t in range(self.time_steps - 1, 0, -1):  # Reversed from T to 1
                eps_hat = self.backbone(
                    z_current,
                    make_t_batched(t, n, y_embed),
                    y_embed,
                )
                if self.embed_full_choir:
                    z_prev_hat = (1 / (torch.sqrt(1 - self.beta[t]))) * z_current[
                        ..., -1
                    ][..., None] - (
                        self.beta[t]
                        / (torch.sqrt(1 - self.alpha[t]) * torch.sqrt(1 - self.beta[t]))
                    ) * eps_hat
                    eps = torch.randn_like(z_current[..., -1][..., None])
                    z_current = torch.cat(
                        (y[..., 0][..., None], z_prev_hat + eps * self.sigma[t]), dim=-1
                    )
                else:
                    z_prev_hat = (1 / (torch.sqrt(1 - self.beta[t]))) * z_current - (
                        self.beta[t]
                        / (torch.sqrt(1 - self.alpha[t]) * torch.sqrt(1 - self.beta[t]))
                    ) * eps_hat
                    eps = torch.randn_like(z_current)
                    z_current = z_prev_hat + eps * self.sigma[t]
                pbar.update()
            # Now for z_0:
            eps_hat = self.backbone(z_current, make_t_batched(t, n, y_embed), y_embed)
            if self.embed_full_choir:
                x_hat = (1 / (torch.sqrt(1 - self.beta[0]))) * z_current[..., -1][
                    ..., None
                ] - (
                    self.beta[0]
                    / (torch.sqrt(1 - self.alpha[0]) * torch.sqrt(1 - self.beta[0]))
                ) * eps_hat
                x_hat = torch.cat((y[..., 0][..., None], x_hat), dim=-1)
            else:
                x_hat = (1 / (torch.sqrt(1 - self.beta[0]))) * z_current - (
                    self.beta[0]
                    / (torch.sqrt(1 - self.alpha[0]) * torch.sqrt(1 - self.beta[0]))
                ) * eps_hat
            pbar.update()
            pbar.close()
            output = x_hat.view(*_in_shape)
            # TODO: This stuff if backbone is 3d_contact_unet
            # diag_indices = [0, 4, 8]  # Indices of the diagonal elements of the lower triangular matrix
            # gaussian_params[..., diag_indices] = torch.relu(gaussian_params[..., diag_indices]) # Diagonal elements are non-negative
            # print(f"Relu'ed Gaussian params: {gaussian_params.shape}")
            return output


class KPDiffusionModel(torch.nn.Module):
    def __init__(
        self,
        backbone: str,
        time_steps: int,
        beta_1: float,
        beta_T: float,
        n_hand_keypoints: int,  # Will be x3 for x,y,z
        n_obj_keypoints: int,  # Will be x3 for x,y,z
        temporal_dim: int,
        rescale_input: bool,
        y_input_keypoints: Optional[int] = None,  # Will be x3 for x,y,z
        y_embed_dim: Optional[int] = None,
        object_in_encoder: bool = False,
        skip_connections: bool = False,
    ):
        super().__init__()
        self.object_in_encoder = object_in_encoder
        self.n_obj_keypoints = n_obj_keypoints if object_in_encoder else 0
        # TODO: Move all this crap to config (god I'm lazy to do that)
        backbones = {
            "mlp_resnet": (
                partial(
                    MLPResNetBackboneModel,
                    input_dim=(
                        n_hand_keypoints
                        + (self.n_obj_keypoints if object_in_encoder else 0)
                    )
                    * 3,
                    output_dim=n_hand_keypoints * 3,
                    hidden_dim=512,
                    context_dim=y_embed_dim,
                    skip_connections=skip_connections,
                ),
                partial(
                    MLPResNetEncoderModel,
                    input_dim=y_input_keypoints * 3,
                    hidden_dim=2048,
                    embed_dim=y_embed_dim,
                )
                if (y_embed_dim is not None and y_input_keypoints is not None)
                else None,
            ),
            "mlp_resnet_w_pointnet2": (
                partial(
                    MLPResNetBackboneModel,
                    input_dim=(
                        n_hand_keypoints
                        + (self.n_obj_keypoints if object_in_encoder else 0)
                    )
                    * 3,
                    output_dim=n_hand_keypoints * 3,
                    hidden_dim=512,
                    context_dim=y_embed_dim,
                    skip_connections=skip_connections,
                ),
                partial(
                    PointNet2EncoderModel,
                    input_points=y_input_keypoints,
                    embed_dim=y_embed_dim,
                )
                if (y_embed_dim is not None and y_input_keypoints is not None)
                else None,
            ),
        }
        assert backbone in backbones, f"Unknown backbone {backbone}"
        self.backbone = backbones[backbone][0](
            time_encoder=SinusoidalPosEncoder(time_steps, temporal_dim),
            temporal_dim=temporal_dim,
        )
        self.embedder = (
            backbones[backbone][1]()
            if (y_embed_dim is not None and y_input_keypoints is not None)
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
        self.x_hand_mean, self.x_obj_mean, self.x_hand_std, self.x_obj_std = (
            None,
            None,
            None,
            None,
        )
        self.y_hand_mean, self.y_obj_mean, self.y_hand_std, self.y_obj_std = (
            None,
            None,
            None,
            None,
        )

    def set_dataset_stats(
        self,
        dataset: torch.utils.data.Dataset,
    ) -> None:
        self.x_hand_mean, self.x_obj_mean, self.x_hand_std, self.x_obj_std = (
            dataset.gt_kp_hand_mean,
            dataset.gt_kp_obj_mean,
            dataset.gt_kp_hand_std,
            dataset.gt_kp_obj_std,
        )
        self.y_hand_mean, self.y_obj_mean, self.y_hand_std, self.y_obj_std = (
            dataset.noisy_kp_hand_mean,
            dataset.noisy_kp_obj_mean,
            dataset.noisy_kp_hand_std,
            dataset.noisy_kp_obj_std,
        )

    def _standardize(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        return (x - mean.to(x.device)) / std.to(x.device)

    def _destandardize(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        return x * std.to(x.device) + mean.to(x.device)

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._input_shape is None:
            self._input_shape = x.shape[1:]
        assert self.x_hand_mean is not None, "Must call set_dataset_stats first"
        if self.object_in_encoder:
            x[..., : self.n_obj_keypoints, :] = self._standardize(
                x[..., : self.n_obj_keypoints, :], self.x_obj_mean, self.x_obj_std
            )
            x[..., self.n_obj_keypoints :, :] = self._standardize(
                (x[..., self.n_obj_keypoints :, :], self.x_hand_mean, self.x_hand_std)
            )
        else:
            x = self._standardize(x, self.x_hand_mean, self.x_hand_std)
        y[..., : self.n_obj_keypoints, :] = self._standardize(
            y[..., : self.n_obj_keypoints, :], self.y_obj_mean, self.y_obj_std
        )
        y[..., self.n_obj_keypoints :, :] = self._standardize(
            y[..., self.n_obj_keypoints :, :], self.y_hand_mean, self.y_hand_std
        )
        # ===== Training =========
        # 1. Sample timestep t with shape (B, 1)
        t = (
            torch.randint(0, self.time_steps, (x.shape[0], 1))
            .to(x.device)
            .requires_grad_(False)
        )
        # 2. Sample the noise with shape x.shape
        if self.object_in_encoder:
            obj_kpts = x[..., : self.n_obj_keypoints, :]
            x = x[..., self.n_obj_keypoints :, :]
        eps = torch.randn_like(x).to(x.device).requires_grad_(False)
        # print(f"eps.shape: {eps.shape}")
        # 3. Diffuse the image
        batched_alpha = self.alpha[t]
        diffused_x = (
            torch.sqrt(batched_alpha)[..., None] * x
            + torch.sqrt(1 - batched_alpha)[..., None] * eps
        )
        # print(f"diffused_x.shape: {diffused_x.shape}")
        # 4. Predict the noise sample
        y_embed = self.embedder(y) if y is not None else None
        # print(f"Y embed shape: {y_embed.shape if y_embed is not None else None}")
        eps_hat = self.backbone(
            torch.cat([obj_kpts, diffused_x], dim=-2)
            if self.object_in_encoder
            else diffused_x,
            t,
            y_embed,
            debug=False,
        )
        # print(f"eps_hat.shape: {eps_hat.shape}")
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

        # ==== Preprocessing ====
        if y is not None:
            y[..., : self.n_obj_keypoints, :] = self._standardize(
                y[..., : self.n_obj_keypoints, :], self.y_obj_mean, self.y_obj_std
            )
            y[..., self.n_obj_keypoints :, :] = self._standardize(
                y[..., self.n_obj_keypoints :, :], self.y_hand_mean, self.y_hand_std
            )
        # ===== Inference =======
        if self.object_in_encoder:
            # TODO: This is a hack. In the future the object will always be in y I think
            obj_kpts = y[..., : self.n_obj_keypoints, :]
            assert (
                obj_kpts is not None
            ), "Must provide obj_kpts when object_in_encoder is True"
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
                    torch.cat([obj_kpts, z_current], dim=-2)
                    if self.object_in_encoder
                    else z_current,
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
            eps_hat = self.backbone(
                torch.cat([obj_kpts, z_current], dim=-2)
                if self.object_in_encoder
                else z_current,
                make_t_batched(t, n, y_embed),
                y_embed,
            )
            x_hat = (1 / (torch.sqrt(1 - self.beta[0]))) * z_current - (
                self.beta[0]
                / (torch.sqrt(1 - self.alpha[0]) * torch.sqrt(1 - self.beta[0]))
            ) * eps_hat
            pbar.update()
            pbar.close()
            # ====== Postprocessing ======
            output = x_hat.view(*_in_shape)
            output = self._destandardize(output, self.x_hand_mean, self.x_hand_std)
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
        self.diffusion_model = BPSDiffusionModel(backbone, time_steps, beta_1, beta_T)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        latent_map = self.encoder(x)
        return self.diffusion_model(latent_map)

    def generate(self, n: int) -> torch.Tensor:
        return self.decoder(self.diffusion_model.generate(n))
