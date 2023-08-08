#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Core contribution method: the Aggregate Data-driven Prior Variational Encoder-Decoder.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


class Aggregate_VED(torch.nn.Module):
    def __init__(
        self,
        bps_dim: int,
        encoder_layer_dims: Tuple[int],
        decoder_layer_dims: Tuple[int],
        latent_dim: int,
        predict_mano: bool,
        predict_anchor_orientation: bool,
        share_decoder_for_all_tasks: bool,
        remapped_bps_distances: bool,
        batch_norm: bool,
        decoder_use_obj: bool,
        skip_connections: bool,
        residual_connections: bool,
        encoder_dropout: bool,
        decoder_dropout: bool,
        predict_residuals: bool,
    ) -> None:
        super().__init__()
        self.choir_dim = 2  # 0: closest object point distance, 1: fixed anchor distance
        self.latent_dim = latent_dim
        self.use_batch_norm = batch_norm
        self.decoder_use_obj = decoder_use_obj
        self.remapped_bps_distances = remapped_bps_distances
        self.skip_connections = skip_connections
        self.encoder_dropout = encoder_dropout
        self.decoder_dropout = decoder_dropout
        self.predict_residuals = predict_residuals
        self.residual_connections = residual_connections

        assert not (
            predict_residuals and decoder_use_obj
        ), "Cannot predict residuals and use object points as input to the decoder"

        encoder_all_same_dim = all(
            [
                encoder_layer_dims[i] == encoder_layer_dims[i + 1]
                for i in range(len(encoder_layer_dims) - 1)
            ]
        )
        decoder_all_same_dim = all(
            [
                decoder_layer_dims[i] == decoder_layer_dims[i + 1]
                for i in range(len(decoder_layer_dims) - 1)
            ]
        )

        if (decoder_use_obj or predict_residuals) and not decoder_all_same_dim:
            decoder_layer_dims = [bps_dim] * (len(decoder_layer_dims))
        # ======================= Encoder =======================
        # encoder_proj: List[torch.nn.Module] = [torch.nn.Linear(1024, 1024)]
        posterior_encoder: List[torch.nn.Module] = [
            torch.nn.Linear(
                self.choir_dim * bps_dim + (self.choir_dim // 2 * bps_dim),
                encoder_layer_dims[0],
            ),
        ]
        posterior_bn: List[torch.nn.Module] = [
            torch.nn.BatchNorm1d(encoder_layer_dims[0]),
        ]
        for i in range(len(encoder_layer_dims) - 1):
            posterior_encoder.append(
                torch.nn.Linear(encoder_layer_dims[i], encoder_layer_dims[i + 1])
            )
            posterior_bn.append(torch.nn.BatchNorm1d(encoder_layer_dims[i + 1]))
            # if i > 1:
            # encoder_proj.append(
            # torch.nn.Linear(encoder_layer_dims[i], encoder_layer_dims[i + 1]) if not encoder_all_same_dim else torch.nn.Identity()
            # )  # for residual connections
        posterior_encoder.append(
            torch.nn.Linear(encoder_layer_dims[-1], latent_dim * 2)
        )
        self.posterior_encoder = torch.nn.ModuleList(posterior_encoder)
        self.posterior_bn = torch.nn.ModuleList(posterior_bn)
        # self.encoder_proj = torch.nn.Linear
        # For the prior encoder, skip the first linear layer because we only have one CHOIR input
        prior_encoder: List[torch.nn.Module] = [
            torch.nn.Linear(self.choir_dim * bps_dim, encoder_layer_dims[1]),
        ]
        prior_bn: List[torch.nn.Module] = [
            torch.nn.BatchNorm1d(encoder_layer_dims[1]),
        ]
        for i in range(1, len(encoder_layer_dims) - 1):
            prior_encoder.append(
                torch.nn.Linear(encoder_layer_dims[i], encoder_layer_dims[i + 1])
            )
            prior_bn.append(torch.nn.BatchNorm1d(encoder_layer_dims[i + 1]))
        prior_encoder.append(torch.nn.Linear(encoder_layer_dims[-1], latent_dim * 2))
        self.prior_encoder = torch.nn.ModuleList(prior_encoder)
        self.prior_bn = torch.nn.ModuleList(prior_bn)
        # ========================================================
        # ======================= Decoder =======================
        decoder: List[torch.nn.Module] = [
            torch.nn.Linear(
                latent_dim
                + (bps_dim if decoder_use_obj else 0)
                + (bps_dim * 2 if predict_residuals else 0),
                decoder_layer_dims[0],
            ),
        ]
        decoder_bn: List[torch.nn.Module] = [
            torch.nn.BatchNorm1d(decoder_layer_dims[0])
        ]
        z_linear: List[torch.nn.Module] = [torch.nn.Identity()]
        for i in range(len(decoder_layer_dims) - 1):
            decoder.append(
                torch.nn.Linear(decoder_layer_dims[i], decoder_layer_dims[i + 1])
            )
            decoder_bn.append(torch.nn.BatchNorm1d(decoder_layer_dims[i + 1]))
            z_linear.append(torch.nn.Linear(latent_dim, decoder_layer_dims[i + 1]))
        decoder.append(torch.nn.Linear(decoder_layer_dims[-1], bps_dim))
        self.decoder = torch.nn.ModuleList(decoder)
        self.decoder_bn = torch.nn.ModuleList(decoder_bn)
        self.decoder_all_same_dim = decoder_all_same_dim
        if decoder_all_same_dim:
            self.z_linear = torch.nn.Linear(latent_dim, decoder_layer_dims[0])
        else:
            self.z_linear = torch.nn.ModuleList(z_linear)
        # ========================================================
        # ======================= MANO ===========================
        self.mano_decoder, self.mano_params_decoder, self.mano_pose_decoder = (
            None,
            None,
            None,
        )
        if predict_mano:
            mano_decoder: List[torch.nn.Module] = [
                torch.nn.Linear(
                    latent_dim + (bps_dim if decoder_use_obj else 0),
                    decoder_layer_dims[0],
                ),
                torch.nn.BatchNorm1d(decoder_layer_dims[0])
                if batch_norm
                else torch.nn.Identity(),
                torch.nn.ReLU(),
            ]
            for i in range(len(decoder_layer_dims) - 1):
                mano_decoder.append(
                    torch.nn.Linear(decoder_layer_dims[i], decoder_layer_dims[i + 1])
                )
                if batch_norm:
                    mano_decoder.append(torch.nn.BatchNorm1d(decoder_layer_dims[i + 1]))
                mano_decoder.append(torch.nn.ReLU())
            self.mano_decoder = torch.nn.Sequential(*mano_decoder)
            self.mano_params_decoder = torch.nn.Linear(decoder_layer_dims[-1], 10 + 18)
            self.mano_pose_decoder = torch.nn.Linear(decoder_layer_dims[-1], 6 + 3)

    def forward(
        self, x: torch.Tensor, y: Optional[torch.Tensor] = None, use_mean: bool = False
    ) -> torch.Tensor:
        input_shape = x.shape
        B, T, P, _ = input_shape
        _x = x

        # ========================= Prior distribution =========================
        x = _x.flatten(start_dim=2)
        x_prev = x
        for i, (layer, bn) in enumerate(zip(self.prior_encoder, self.prior_bn)):
            x = layer(x)
            if self.use_batch_norm:
                x = bn(x.view(B * T, -1)).view(B, T, -1)
            if self.residual_connections and i > 0:
                x -= x_prev  # self.encoder_proj[i](x_prev)
            if self.encoder_dropout:
                x = F.dropout(x, p=0.1, training=self.training)
            x = F.relu(x, inplace=False)
            x_prev = x
        if self.encoder_dropout:
            x = F.dropout(x, p=0.1, training=self.training)
        s = self.prior_encoder[-1](x)
        s_c = torch.mean(s, dim=1)
        z_mu, z_log_var = torch.split(s_c, self.latent_dim, dim=-1)
        z_var = 0.1 + 0.9 * torch.sigmoid(z_log_var)
        prior, posterior = torch.distributions.Normal(z_mu, z_var), None

        if y is None:
            # Sample from the prior latent distribution q(z|O) at test time, with O being the
            # noisy observations. rsample() uses the reparameterization trick, allowing us to
            # differentiate through z and into the latent encoder layers.
            z = z_mu if use_mean else prior.rsample()
        else:
            # ======================= Posterior distribution =======================
            # Sample from the posterior latent distribution p(z|O, T) at training time, with O
            # being the noisy observations and P being the ground-truth targets.  This gives us P
            # so that we can learn the approximative Q with KL Divergence, which we then use as a
            # prior at test time when we don't have access to the targets.
            # x = torch.concat([_x.flatten(start_dim=2), y.flatten(start_dim=2)], dim=-1)
            x = torch.cat([_x, y[..., -1].unsqueeze(-1)], dim=-1).flatten(start_dim=2)
            x_prev = x
            for i, (layer, bn) in enumerate(
                zip(self.posterior_encoder, self.posterior_bn)
            ):
                x = layer(x)
                if self.use_batch_norm:
                    x = bn(x.view(B * T, -1)).view(B, T, -1)
                if self.residual_connections and i > 2:
                    x -= x_prev  # self.encoder_proj[i](x_prev)
                x = F.relu(x, inplace=False)
                x_prev = x
            # if self.use_dropout:
            # x = F.dropout(x, p=0.5, training=self.training)
            s = self.posterior_encoder[-1](x)
            s_c = torch.mean(s, dim=1)
            p_mu, p_log_var = torch.split(s_c, self.latent_dim, dim=-1)
            p_var = 0.1 + 0.9 * torch.sigmoid(p_log_var)
            posterior = torch.distributions.Normal(p_mu, p_var)
            z = p_mu if use_mean else posterior.rsample()

        if self.decoder_use_obj:
            dec_input = torch.cat([z, _x[:, 0, :, 0]], dim=-1)
        elif self.predict_residuals:
            z = z.unsqueeze(1).repeat(1, T, 1)
            dec_input = torch.cat([z, _x.flatten(start_dim=2)], dim=-1)
        else:
            dec_input = z

        if self.decoder_all_same_dim:
            z_linears = [self.z_linear] * len(self.decoder)
        else:
            z_linears = self.z_linear

        x = dec_input
        for i, (layer, z_layer, bn) in enumerate(
            zip(self.decoder, z_linears, self.decoder_bn)
        ):
            x = layer(x)
            if self.use_batch_norm:
                if self.predict_residuals:
                    x = bn(x.view(B * T, -1)).view(B, T, -1)
                else:
                    x = bn(x)
            if (
                self.skip_connections and i > 0
            ):  # Skip the first layer since Z is already the input
                x += z_layer(z)
            if self.decoder_dropout:
                x = F.dropout(x, p=0.1, training=self.training)
            x = F.relu(x, inplace=False)
        if self.decoder_dropout:
            x = F.dropout(x, p=0.1, training=self.training)
        sqrt_distances = self.decoder[-1](x)
        if self.remapped_bps_distances:
            # No need to square the distances since we're using sigmoid
            anchor_dist = torch.sigmoid(sqrt_distances)
        else:
            anchor_dist = sqrt_distances**2

        if self.predict_residuals:
            anchor_dist = anchor_dist.view(B, T, P, 1)  # N, T, P, 1
        else:
            anchor_dist = anchor_dist.view(B, P, 1)  # N, 1

        mano = None
        if self.mano_params_decoder is not None and self.mano_pose_decoder is not None:
            mano_embedding = self.mano_decoder(dec_input)
            mano_params = self.mano_params_decoder(mano_embedding)
            mano_pose = self.mano_pose_decoder(mano_embedding)
            mano = torch.cat([mano_params, mano_pose], dim=-1)

        if self.predict_residuals:
            choir = torch.cat(
                [
                    _x.view(B, T, P, self.choir_dim)[..., 0].unsqueeze(-1),
                    _x.view(B, T, P, self.choir_dim)[..., 1].unsqueeze(-1)
                    - anchor_dist,  # We're predicting the "residuals" from the anchor distance
                ],
                dim=-1,
            )
        else:
            choir = torch.cat(
                [
                    _x.view(B, T, P, self.choir_dim)[
                        :, 0, :, 0
                    ]  # Use the first view since the object BPS isn't noisy
                    .unsqueeze(-1)
                    .requires_grad_(False),
                    anchor_dist,
                ],
                dim=-1,
            )
        return {
            "choir": choir,
            "orientations": None,
            "mano": mano,
            "prior": prior,
            "posterior": posterior,
        }


class Aggregate_VED_Simple(torch.nn.Module):
    def __init__(
        self,
        bps_dim: int,
        encoder_layer_dims: Tuple[int],
        decoder_layer_dims: Tuple[int],
        latent_dim: int,
        predict_mano: bool,
        predict_anchor_orientation: bool,
        share_decoder_for_all_tasks: bool,
        remapped_bps_distances: bool,
        batch_norm: bool,
        decoder_use_obj: bool,
        skip_connections: bool,
        use_dropout: bool,
    ) -> None:
        super().__init__()
        self.choir_dim = 2  # 0: closest object point distance, 1: fixed anchor distance
        self.latent_dim = latent_dim
        self.use_batch_norm = batch_norm
        self.decoder_use_obj = decoder_use_obj
        self.remapped_bps_distances = remapped_bps_distances
        self.skip_connections = skip_connections
        self.use_dropout = use_dropout

        encoder_all_same_dim = all(
            [
                encoder_layer_dims[i] == encoder_layer_dims[i + 1]
                for i in range(len(encoder_layer_dims) - 1)
            ]
        )
        decoder_all_same_dim = all(
            [
                decoder_layer_dims[i] == decoder_layer_dims[i + 1]
                for i in range(len(decoder_layer_dims) - 1)
            ]
        )

        if decoder_use_obj and not decoder_all_same_dim:
            decoder_layer_dims = [bps_dim] * (len(decoder_layer_dims))
        # ======================= Encoder =======================
        encoder_proj: List[torch.nn.Module] = [
            torch.nn.Identity(),
        ]
        encoder: List[torch.nn.Module] = [
            torch.nn.Linear(self.choir_dim * bps_dim, encoder_layer_dims[0]),
        ]
        encoder_bn: List[torch.nn.Module] = [
            torch.nn.BatchNorm1d(encoder_layer_dims[0]),
        ]
        for i in range(len(encoder_layer_dims) - 1):
            encoder.append(
                torch.nn.Linear(encoder_layer_dims[i], encoder_layer_dims[i + 1])
            )
            encoder_bn.append(torch.nn.BatchNorm1d(encoder_layer_dims[i + 1]))
            encoder_proj.append(
                torch.nn.Linear(encoder_layer_dims[i], encoder_layer_dims[i + 1])
                if not encoder_all_same_dim
                else torch.nn.Identity()
            )  # for residual connections
        encoder.append(torch.nn.Linear(encoder_layer_dims[-1], latent_dim * 2))
        self.encoder = torch.nn.ModuleList(encoder)
        self.encoder_bn = torch.nn.ModuleList(encoder_bn)
        self.encoder_proj = torch.nn.ModuleList(encoder_proj)
        # ========================================================
        # ======================= Decoder =======================
        decoder: List[torch.nn.Module] = [
            torch.nn.Linear(
                latent_dim + (bps_dim if decoder_use_obj else 0), decoder_layer_dims[0]
            ),
        ]
        decoder_bn: List[torch.nn.Module] = [
            torch.nn.BatchNorm1d(decoder_layer_dims[0])
        ]
        z_linear: List[torch.nn.Module] = [torch.nn.Identity()]
        for i in range(len(decoder_layer_dims) - 1):
            decoder.append(
                torch.nn.Linear(decoder_layer_dims[i], decoder_layer_dims[i + 1])
            )
            decoder_bn.append(torch.nn.BatchNorm1d(decoder_layer_dims[i + 1]))
            z_linear.append(torch.nn.Linear(latent_dim, decoder_layer_dims[i + 1]))
        decoder.append(torch.nn.Linear(decoder_layer_dims[-1], bps_dim))
        self.decoder = torch.nn.ModuleList(decoder)
        self.decoder_bn = torch.nn.ModuleList(decoder_bn)
        self.decoder_all_same_dim = decoder_all_same_dim
        if decoder_all_same_dim:
            self.z_linear = torch.nn.Linear(latent_dim, decoder_layer_dims[0])
        else:
            self.z_linear = torch.nn.ModuleList(z_linear)
        # ========================================================
        # ======================= MANO ===========================
        self.mano_decoder, self.mano_params_decoder, self.mano_pose_decoder = (
            None,
            None,
            None,
        )
        if predict_mano:
            mano_decoder: List[torch.nn.Module] = [
                torch.nn.Linear(
                    latent_dim + (bps_dim if decoder_use_obj else 0),
                    decoder_layer_dims[0],
                ),
                torch.nn.BatchNorm1d(decoder_layer_dims[0])
                if batch_norm
                else torch.nn.Identity(),
                torch.nn.ReLU(),
            ]
            for i in range(len(decoder_layer_dims) - 1):
                mano_decoder.append(
                    torch.nn.Linear(decoder_layer_dims[i], decoder_layer_dims[i + 1])
                )
                if batch_norm:
                    mano_decoder.append(torch.nn.BatchNorm1d(decoder_layer_dims[i + 1]))
                mano_decoder.append(torch.nn.ReLU())
            self.mano_decoder = torch.nn.Sequential(*mano_decoder)
            self.mano_params_decoder = torch.nn.Linear(decoder_layer_dims[-1], 10 + 18)
            self.mano_pose_decoder = torch.nn.Linear(decoder_layer_dims[-1], 6 + 3)

    def forward(
        self, x: torch.Tensor, y: Optional[torch.Tensor] = None, use_mean: bool = False
    ) -> torch.Tensor:
        input_shape = x.shape
        B, T, P, _ = input_shape
        _x = x

        # ========================= Prior distribution =========================
        x = _x.flatten(start_dim=2)
        x_prev = x
        for i, (layer, proj_layer, bn) in enumerate(
            zip(self.encoder, self.encoder_proj, self.encoder_bn)
        ):
            x = layer(x)
            if self.use_batch_norm:
                x = bn(x.view(B * T, -1)).view(B, T, -1)
            if self.skip_connections and i > 0:
                x -= proj_layer(x_prev)
            x = F.relu(x, inplace=False)
            x_prev = x
        if self.use_dropout:
            x = F.dropout(x, p=0.5, training=self.training)
        s = self.encoder[-1](x)
        s_c = torch.mean(s, dim=1)
        z_mu, z_log_var = torch.split(s_c, self.latent_dim, dim=-1)
        # latent_sigma = 0.1 + 0.9 * torch.sigmoid(latent_sigma_sorta)
        z_var = torch.exp(z_log_var)
        prior, posterior = torch.distributions.Normal(z_mu, z_var), None

        if y is None:
            # Sample from the prior latent distribution q(z|O) at test time, with O being the
            # noisy observations. rsample() uses the reparameterization trick, allowing us to
            # differentiate through z and into the latent encoder layers.
            z = z_mu if use_mean else prior.rsample()
        else:
            # ======================= Posterior distribution =======================
            # Sample from the posterior latent distribution p(z|O, T) at training time, with O
            # being the noisy observations and P being the ground-truth targets.  This gives us P
            # so that we can learn the approximative Q with KL Divergence, which we then use as a
            # prior at test time when we don't have access to the targets.
            # x = torch.concat([_x.flatten(start_dim=2), y.flatten(start_dim=2)], dim=-1)
            x = y.flatten(start_dim=2)
            x_prev = x
            for i, (layer, proj_layer, bn) in enumerate(
                zip(self.encoder, self.encoder_proj, self.encoder_bn)
            ):
                x = layer(x)
                if self.use_batch_norm:
                    x = bn(x.view(B * T, -1)).view(B, T, -1)
                if self.skip_connections and i > 0:
                    x -= proj_layer(x_prev)
                x = F.relu(x, inplace=False)
                x_prev = x
            if self.use_dropout:
                x = F.dropout(x, p=0.5, training=self.training)
            s = self.encoder[-1](x)
            s_c = torch.mean(s, dim=1)
            p_mu, p_log_var = torch.split(s_c, self.latent_dim, dim=-1)
            # latent_sigma = 0.1 + 0.9 * torch.sigmoid(latent_sigma_sorta)
            p_var = torch.exp(p_log_var)
            posterior = torch.distributions.Normal(p_mu, p_var)
            z = p_mu if use_mean else posterior.rsample()

        dec_input = (
            z if not self.decoder_use_obj else torch.cat([z, _x[:, 0, :, 0]], dim=-1)
        )
        x = dec_input
        # sqrt_distances = self.decoder(dec_input)
        if self.decoder_all_same_dim:
            z_linears = [self.z_linear] * len(self.decoder)
        else:
            z_linears = self.z_linear
        for i, (layer, z_layer, bn) in enumerate(
            zip(self.decoder, z_linears, self.decoder_bn)
        ):
            x = layer(x)
            if self.use_batch_norm:
                x = bn(x)
            if (
                self.skip_connections and i > 0
            ):  # Skip the first layer since Z is already the input
                x += z_layer(z)
            x = F.relu(x, inplace=False)
        sqrt_distances = self.decoder[-1](x)
        if self.remapped_bps_distances:
            # No need to square the distances since we're using sigmoid
            anchor_dist = (torch.sigmoid(sqrt_distances)).view(B, P, 1)  # N, 1
        else:
            anchor_dist = (sqrt_distances**2).view(B, P, 1)  # N, 1

        mano = None
        if self.mano_params_decoder is not None and self.mano_pose_decoder is not None:
            mano_embedding = self.mano_decoder(dec_input)
            mano_params = self.mano_params_decoder(mano_embedding)
            mano_pose = self.mano_pose_decoder(mano_embedding)
            mano = torch.cat([mano_params, mano_pose], dim=-1)

        choir = torch.cat(
            [
                _x.view(B, T, P, self.choir_dim)[
                    :, 0, :, 0
                ]  # Use the first view since the object BPS isn't noisy
                .unsqueeze(-1)
                .requires_grad_(False),
                anchor_dist,
            ],
            dim=-1,
        )
        return {
            "choir": choir,
            "orientations": None,
            "mano": mano,
            "prior": prior,
            "posterior": posterior,
        }


class Aggregate_VED_SuperSimple(torch.nn.Module):
    def __init__(
        self,
        bps_dim: int,
        encoder_layer_dims: Tuple[int],
        decoder_layer_dims: Tuple[int],
        latent_dim: int,
        predict_mano: bool,
        predict_anchor_orientation: bool,
        share_decoder_for_all_tasks: bool,
        remapped_bps_distances: bool,
        batch_norm: bool,
        decoder_use_obj: bool,
        skip_connections: bool,
        use_dropout: bool,
    ) -> None:
        super().__init__()
        self.choir_dim = 2  # 0: closest object point distance, 1: fixed anchor distance
        self.latent_dim = latent_dim
        self.use_batch_norm = batch_norm
        self.decoder_use_obj = decoder_use_obj
        self.remapped_bps_distances = remapped_bps_distances
        self.skip_connections = skip_connections
        self.use_dropout = use_dropout

        encoder_all_same_dim = all(
            [
                encoder_layer_dims[i] == encoder_layer_dims[i + 1]
                for i in range(len(encoder_layer_dims) - 1)
            ]
        )
        decoder_all_same_dim = all(
            [
                decoder_layer_dims[i] == decoder_layer_dims[i + 1]
                for i in range(len(decoder_layer_dims) - 1)
            ]
        )

        if decoder_use_obj and not decoder_all_same_dim:
            decoder_layer_dims = [bps_dim] * (len(decoder_layer_dims))
        # ======================= Encoder =======================
        encoder_proj: List[torch.nn.Module] = [
            torch.nn.Identity(),
        ]
        encoder: List[torch.nn.Module] = [
            torch.nn.Linear(self.choir_dim * bps_dim, encoder_layer_dims[0]),
        ]
        encoder_bn: List[torch.nn.Module] = [
            torch.nn.BatchNorm1d(encoder_layer_dims[0]),
        ]
        for i in range(len(encoder_layer_dims) - 1):
            encoder.append(
                torch.nn.Linear(encoder_layer_dims[i], encoder_layer_dims[i + 1])
            )
            encoder_bn.append(torch.nn.BatchNorm1d(encoder_layer_dims[i + 1]))
            encoder_proj.append(
                torch.nn.Linear(encoder_layer_dims[i], encoder_layer_dims[i + 1])
                if not encoder_all_same_dim
                else torch.nn.Identity()
            )  # for residual connections
        encoder.append(torch.nn.Linear(encoder_layer_dims[-1], latent_dim))
        self.encoder = torch.nn.ModuleList(encoder)
        self.encoder_bn = torch.nn.ModuleList(encoder_bn)
        self.encoder_proj = torch.nn.ModuleList(encoder_proj)
        # ========================================================
        # ======================= Decoder =======================
        decoder: List[torch.nn.Module] = [
            torch.nn.Linear(
                latent_dim + (bps_dim if decoder_use_obj else 0), decoder_layer_dims[0]
            ),
        ]
        decoder_bn: List[torch.nn.Module] = [
            torch.nn.BatchNorm1d(decoder_layer_dims[0])
        ]
        z_linear: List[torch.nn.Module] = [torch.nn.Identity()]
        for i in range(len(decoder_layer_dims) - 1):
            decoder.append(
                torch.nn.Linear(decoder_layer_dims[i], decoder_layer_dims[i + 1])
            )
            decoder_bn.append(torch.nn.BatchNorm1d(decoder_layer_dims[i + 1]))
            z_linear.append(torch.nn.Linear(latent_dim, decoder_layer_dims[i + 1]))
        decoder.append(torch.nn.Linear(decoder_layer_dims[-1], bps_dim))
        self.decoder = torch.nn.ModuleList(decoder)
        self.decoder_bn = torch.nn.ModuleList(decoder_bn)
        self.decoder_all_same_dim = decoder_all_same_dim
        if decoder_all_same_dim:
            self.z_linear = torch.nn.Linear(latent_dim, decoder_layer_dims[0])
        else:
            self.z_linear = torch.nn.ModuleList(z_linear)
        # ========================================================
        # ======================= MANO ===========================
        self.mano_decoder, self.mano_params_decoder, self.mano_pose_decoder = (
            None,
            None,
            None,
        )
        if predict_mano:
            mano_decoder: List[torch.nn.Module] = [
                torch.nn.Linear(
                    latent_dim + (bps_dim if decoder_use_obj else 0),
                    decoder_layer_dims[0],
                ),
                torch.nn.BatchNorm1d(decoder_layer_dims[0])
                if batch_norm
                else torch.nn.Identity(),
                torch.nn.ReLU(),
            ]
            for i in range(len(decoder_layer_dims) - 1):
                mano_decoder.append(
                    torch.nn.Linear(decoder_layer_dims[i], decoder_layer_dims[i + 1])
                )
                if batch_norm:
                    mano_decoder.append(torch.nn.BatchNorm1d(decoder_layer_dims[i + 1]))
                mano_decoder.append(torch.nn.ReLU())
            self.mano_decoder = torch.nn.Sequential(*mano_decoder)
            self.mano_params_decoder = torch.nn.Linear(decoder_layer_dims[-1], 10 + 18)
            self.mano_pose_decoder = torch.nn.Linear(decoder_layer_dims[-1], 6 + 3)

    def forward(
        self, x: torch.Tensor, y: Optional[torch.Tensor] = None, use_mean: bool = False
    ) -> torch.Tensor:
        input_shape = x.shape
        B, T, P, _ = input_shape
        _x = x

        x = _x.flatten(start_dim=2)
        x_prev = x
        for i, (layer, proj_layer, bn) in enumerate(
            zip(self.encoder, self.encoder_proj, self.encoder_bn)
        ):
            x = layer(x)
            if self.use_batch_norm:
                x = bn(x.view(B * T, -1)).view(B, T, -1)
            if self.skip_connections and i > 0:
                x -= proj_layer(x_prev)
            x = F.relu(x, inplace=False)
            x_prev = x
        if self.use_dropout:
            x = F.dropout(x, p=0.5, training=self.training)
        s = self.encoder[-1](x)
        z = torch.mean(s, dim=1)

        dec_input = (
            z if not self.decoder_use_obj else torch.cat([z, _x[:, 0, :, 0]], dim=-1)
        )
        x = dec_input
        # sqrt_distances = self.decoder(dec_input)
        if self.decoder_all_same_dim:
            z_linears = [self.z_linear] * len(self.decoder)
        else:
            z_linears = self.z_linear
        for i, (layer, z_layer, bn) in enumerate(
            zip(self.decoder, z_linears, self.decoder_bn)
        ):
            x = layer(x)
            if self.use_batch_norm:
                x = bn(x)
            if (
                self.skip_connections and i > 0
            ):  # Skip the first layer since Z is already the input
                x += z_layer(z)
            x = F.relu(x, inplace=False)
        sqrt_distances = self.decoder[-1](x)
        if self.remapped_bps_distances:
            # No need to square the distances since we're using sigmoid
            anchor_dist = (torch.sigmoid(sqrt_distances)).view(B, P, 1)  # N, 1
        else:
            anchor_dist = (sqrt_distances**2).view(B, P, 1)  # N, 1

        mano = None
        if self.mano_params_decoder is not None and self.mano_pose_decoder is not None:
            mano_embedding = self.mano_decoder(dec_input)
            mano_params = self.mano_params_decoder(mano_embedding)
            mano_pose = self.mano_pose_decoder(mano_embedding)
            mano = torch.cat([mano_params, mano_pose], dim=-1)

        choir = torch.cat(
            [
                _x.view(B, T, P, self.choir_dim)[
                    :, 0, :, 0
                ]  # Use the first view since the object BPS isn't noisy
                .unsqueeze(-1)
                .requires_grad_(False),
                anchor_dist,
            ],
            dim=-1,
        )
        return {
            "choir": choir,
            "orientations": None,
            "mano": mano,
            "prior": None,
            "posterior": None,
        }
