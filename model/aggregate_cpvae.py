#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Core contribution method: the Aggregate Conditional-Prior Variational Auto-Encoder.
"""

from typing import List, Optional, Tuple

import torch


class Aggregate_CPVAE(torch.nn.Module):
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
    ) -> None:
        super().__init__()
        self.choir_dim = 2  # 0: closest object point distance, 1: fixed anchor distance
        self.latent_dim = latent_dim

        # ======================= Encoder =======================
        posterior_encoder: List[torch.nn.Module] = [
            torch.nn.Linear(self.choir_dim * bps_dim * 2, encoder_layer_dims[0]),
            # torch.nn.BatchNorm1d(encoder_layer_dims[0]),
            torch.nn.ReLU(),
        ]
        for i in range(len(encoder_layer_dims) - 1):
            posterior_encoder.append(
                torch.nn.Linear(encoder_layer_dims[i], encoder_layer_dims[i + 1])
            )
            # encoder.append(torch.nn.BatchNorm1d(encoder_layer_dims[i + 1]))
            posterior_encoder.append(torch.nn.ReLU())
        posterior_encoder.append(
            torch.nn.Linear(encoder_layer_dims[-1], latent_dim * 2)
        )
        self.posterior_encoder = torch.nn.Sequential(*posterior_encoder)
        # For the prior encoder, skip the first linear layer because we only have one CHOIR input
        prior_encoder: List[torch.nn.Module] = [
            torch.nn.Linear(self.choir_dim * bps_dim, encoder_layer_dims[1]),
            # torch.nn.BatchNorm1d(encoder_layer_dims[0]),
            torch.nn.ReLU(),
        ]
        for i in range(1, len(encoder_layer_dims) - 1):
            prior_encoder.append(
                torch.nn.Linear(encoder_layer_dims[i], encoder_layer_dims[i + 1])
            )
            # encoder.append(torch.nn.BatchNorm1d(encoder_layer_dims[i + 1]))
            prior_encoder.append(torch.nn.ReLU())
        prior_encoder.append(torch.nn.Linear(encoder_layer_dims[-1], latent_dim * 2))
        self.prior_encoder = torch.nn.Sequential(*prior_encoder)
        # ========================================================
        # ======================= Decoder =======================
        decoder: List[torch.nn.Module] = [
            torch.nn.Linear(latent_dim, decoder_layer_dims[0]),
            # torch.nn.BatchNorm1d(decoder_layer_dims[0]),
            torch.nn.ReLU(),
        ]
        for i in range(len(decoder_layer_dims) - 1):
            decoder.append(
                torch.nn.Linear(decoder_layer_dims[i], decoder_layer_dims[i + 1])
            )
            # decoder.append(torch.nn.BatchNorm1d(decoder_layer_dims[i + 1]))
            decoder.append(torch.nn.ReLU())
        decoder.append(torch.nn.Linear(decoder_layer_dims[-1], bps_dim))
        if remapped_bps_distances:
            decoder.append(torch.nn.Sigmoid())
        self._remapped_bps_distances = remapped_bps_distances
        self.decoder = torch.nn.Sequential(*decoder)
        # ========================================================
        # ======================= MANO ===========================
        self.mano_decoder, self.mano_params_decoder, self.mano_pose_decoder = (
            None,
            None,
            None,
        )
        if predict_mano:
            mano_decoder: List[torch.nn.Module] = [
                torch.nn.Linear(latent_dim, decoder_layer_dims[0]),
                # torch.nn.BatchNorm1d(decoder_layer_dims[0]),
                torch.nn.ReLU(),
            ]
            for i in range(len(decoder_layer_dims) - 1):
                mano_decoder.append(
                    torch.nn.Linear(decoder_layer_dims[i], decoder_layer_dims[i + 1])
                )
                # mano_decoder.append(torch.nn.BatchNorm1d(decoder_layer_dims[i + 1]))
                mano_decoder.append(torch.nn.ReLU())
            # mano_decoder.append(torch.nn.Linear(decoder_layer_dims[-1], 10 + 18))
            self.mano_decoder = torch.nn.Sequential(*mano_decoder)
            self.mano_params_decoder = torch.nn.Linear(decoder_layer_dims[-1], 10 + 18)
            self.mano_pose_decoder = torch.nn.Linear(decoder_layer_dims[-1], 6 + 3)

    def forward(
        self, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        input_shape = x.shape
        B, T, P, _ = input_shape
        _x = x

        # ========================= Prior distribution =========================
        s = self.prior_encoder(x.flatten(start_dim=2))
        s_c = torch.mean(s, dim=1)
        z_mu, z_log_var = torch.split(s_c, self.latent_dim, dim=-1)
        # latent_sigma = 0.1 + 0.9 * torch.sigmoid(latent_sigma_sorta)
        z_var = torch.exp(z_log_var)
        prior, posterior = torch.distributions.Normal(z_mu, z_var), None

        if y is None:
            # Sample from the prior latent distribution q(z|O) at test time, with O being the
            # noisy observations. rsample() uses the reparameterization trick, allowing us to
            # differentiate through z and into the latent encoder layers.
            z = prior.rsample()
        else:
            # ======================= Posterior distribution =======================
            # Sample from the posterior latent distribution p(z|O, T) at training time, with O
            # being the noisy observations and P being the ground-truth targets.  This gives us P
            # so that we can learn the approximative Q with KL Divergence, which we then use as a
            # prior at test time when we don't have access to the targets.
            s = self.posterior_encoder(
                torch.concat([x.flatten(start_dim=2), y.flatten(start_dim=2)], dim=-1)
            )
            s_c = torch.mean(s, dim=1)
            p_mu, p_log_var = torch.split(s_c, self.latent_dim, dim=-1)
            # latent_sigma = 0.1 + 0.9 * torch.sigmoid(latent_sigma_sorta)
            p_var = torch.exp(p_log_var)
            posterior = torch.distributions.Normal(p_mu, p_var)
            z = posterior.rsample()

        sqrt_distances = self.decoder(z)
        if self._remapped_bps_distances:
            # No need to square the distances since we're using sigmoid
            anchor_dist = (sqrt_distances).view(B, P, 1)  # N, 1
        else:
            anchor_dist = (sqrt_distances**2).view(B, P, 1)  # N, 1

        mano = None
        if self.mano_params_decoder is not None and self.mano_pose_decoder is not None:
            mano_embedding = self.mano_decoder(z)
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
