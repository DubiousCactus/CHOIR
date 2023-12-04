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

from model.blocks import AttentionAggregator


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
        predict_deltas: bool,
        frame_to_predict: str = "average",
        aggregator: str = "mean",
        agg_heads: int = 8,
        agg_kq_dim: int = 1024,
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
        self.predict_deltas = predict_deltas
        self.residual_connections = residual_connections
        assert frame_to_predict in [
            "average",
            "last",
        ], "Invalid frame to predict. Must be 'average' or 'last'"
        self.frame_to_predict = frame_to_predict
        assert aggregator in [
            "mean",
            "attention",
            "attention_pytorch",
        ], "Invalid aggregator. Must be 'mean', 'attention' or 'attention_pytorch'"
        # Multi-head cross-attention aggregator
        self.mhca_aggregator = (
            AttentionAggregator(
                "multi_head_pytorch"
                if aggregator == "attention_pytorch"
                else "multi_head",
                multi_head_use_bias=True,
                n_heads=agg_heads,
                k_dim_in=(self.choir_dim - 1)
                * bps_dim,  # -1 cause not including the object
                k_dim_out=bps_dim if aggregator == "attention_pytorch" else agg_kq_dim,
                q_dim_in=(self.choir_dim - 1) * bps_dim,
                q_dim_out=bps_dim if aggregator == "attention_pytorch" else agg_kq_dim,
                v_dim_in=latent_dim * 2,
                v_dim_out=latent_dim * 2,
            )
            if aggregator.startswith("attention")
            else None
        )

        assert not (
            predict_deltas and decoder_use_obj
        ), "Cannot both predict deltas and use object points as input to the decoder"

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

        if (decoder_use_obj or predict_deltas) and not decoder_all_same_dim:
            decoder_layer_dims = [bps_dim * (2 if predict_deltas else 1)] * (
                len(decoder_layer_dims)
            )
        # ======================= Encoder =======================
        # encoder_proj: List[torch.nn.Module] = [torch.nn.Linear(1024, 1024)]
        posterior_encoder: List[torch.nn.Module] = [
            torch.nn.Linear(
                self.choir_dim * bps_dim,
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
            torch.nn.Linear(self.choir_dim * bps_dim, encoder_layer_dims[0]),
        ]
        prior_bn: List[torch.nn.Module] = [
            torch.nn.BatchNorm1d(encoder_layer_dims[0]),
        ]
        for i in range(0, len(encoder_layer_dims) - 1):
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
                + (bps_dim * 2 if predict_deltas or frame_to_predict == "last" else 0),
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
        if self.mhca_aggregator is not None:
            # s_c = torch.mean(s, dim=1)
            # Key: _x[..., 1:].flatten(start_dim=2) / The noisy anchor distances
            # Value: s / The latent distributions
            # Query: _x[:, -1].flatten(start_dim=1) / The last frame's anchor distances
            s_c = self.mhca_aggregator(
                _x[..., 1:].flatten(start_dim=2),
                s,
                _x[:, -1, :, 1:].flatten(start_dim=1).unsqueeze(1),
            ).squeeze()
        else:
            s_c = torch.mean(s, dim=1)
        p_mu, p_log_var = torch.split(s_c, self.latent_dim, dim=-1)
        p_var = 0.1 + 0.9 * torch.sigmoid(p_log_var)
        prior, posterior = torch.distributions.Normal(p_mu, p_var), None

        if y is None:
            # Sample from the prior latent distribution q(z|O) at test time, with O being the
            # noisy observations. rsample() uses the reparameterization trick, allowing us tola
            # differentiate through z and into the latent encoder layers.
            z = p_mu if use_mean else prior.rsample()
            q_mu, q_var = None, None
        else:
            # ======================= Posterior distribution =======================
            # Sample from the posterior latent distribution p(z|T) at training time, with T being
            # the ground-truth targets.  This gives us p(.) so that we can learn the approximative
            # q(.) with KL Divergence, which we then use as a prior at test time when we don't have
            # access to the targets.
            x = y.flatten(start_dim=2)
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
            if self.mhca_aggregator is not None:
                # s_c = torch.mean(s, dim=1)
                # Key: _x[..., 1:].flatten(start_dim=2) / The noisy anchor distances
                # Value: s / The latent distributions
                # Query: _x[:, -1].flatten(start_dim=1) / The last frame's anchor distances
                s_c = self.mhca_aggregator(
                    _x[..., 1:].flatten(start_dim=2),
                    s,
                    _x[:, -1, :, 1:].flatten(start_dim=1).unsqueeze(1),
                ).squeeze()
            else:
                s_c = torch.mean(s, dim=1)
            q_mu, q_log_var = torch.split(s_c, self.latent_dim, dim=-1)
            q_var = 0.1 + 0.9 * torch.sigmoid(q_log_var)
            posterior = torch.distributions.Normal(q_mu, q_var)
            z = q_mu if use_mean else posterior.rsample()

        # When predicting the average frame for contact pose, we just take any frame since they're
        # all the same.
        last_frame = _x[:, -1]
        if self.decoder_use_obj:
            dec_input = torch.cat([z, last_frame[..., 0]], dim=-1)
        elif self.predict_deltas or self.frame_to_predict == "last":
            dec_input = torch.cat([z, last_frame.flatten(start_dim=1)], dim=-1)
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
                x = bn(x)
            if (
                # Skip connections mess everything up once we add a dependency on the field in the
                # decoder, so we only use them for the 2nd layer (the first being fed [z, field])
                self.skip_connections
                and (i > 0 if not (self.predict_deltas) else i in [1, 2])
            ):  # Skip the first layer since Z is already the input
                x += z_layer(z)
            if self.decoder_dropout:
                x = F.dropout(x, p=0.1, training=self.training)
            x = F.relu(x, inplace=False)
        if self.decoder_dropout:
            x = F.dropout(x, p=0.1, training=self.training)
        sqrt_distances = self.decoder[-1](x)
        if self.predict_deltas:
            anchor_deltas = torch.tanh(sqrt_distances)
        elif self.remapped_bps_distances:
            # No need to square the distances since we're using sigmoid
            anchor_dist = torch.sigmoid(sqrt_distances)
        else:
            anchor_dist = sqrt_distances**2

        anchor_dist = anchor_dist.view(B, P, 1)  # N, 1

        mano = None
        if self.mano_params_decoder is not None and self.mano_pose_decoder is not None:
            mano_embedding = self.mano_decoder(dec_input)
            mano_params = self.mano_params_decoder(mano_embedding)
            mano_pose = self.mano_pose_decoder(mano_embedding)
            mano = torch.cat([mano_params, mano_pose], dim=-1)

        if self.predict_deltas:
            if self.remapped_bps_distances:
                anchor_dist = torch.clamp(
                    last_frame[..., -1].unsqueeze(-1) + anchor_deltas, min=1e-12, max=1
                )
                # anchor_dist = torch.relu(last_frame[..., -1].unsqueeze(-1) + anchor_deltas) + 1e-9
                # anchor_dist = _x[:, -1, :, 1].unsqueeze(-1) + anchor_deltas
            else:
                anchor_dist = torch.relu(
                    last_frame[..., -1].unsqueeze(-1) + anchor_deltas
                )
            choir = torch.cat(
                [
                    last_frame[..., 0].unsqueeze(-1),
                    anchor_dist,
                ],
                dim=-1,
            )
        else:
            choir = torch.cat(
                [
                    last_frame[
                        ..., 0
                    ].unsqueeze(  # Use any view since the object BPS isn't noisy
                        -1
                    ),
                    anchor_dist,
                ],
                dim=-1,
            )
        return {
            "choir": choir,
            "orientations": None,
            "mano": mano,
            "prior": (p_mu, p_var),
            "posterior": (q_mu, q_var),
        }
