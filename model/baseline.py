#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Simple baseline to test the data loading pipeline.
"""

from typing import List, Tuple

import torch


class BaselineModel(torch.nn.Module):
    def __init__(
        self,
        bps_dim: int,
        encoder_layer_dims: Tuple[int],
        decoder_layer_dims: Tuple[int],
        latent_dim: int,
        predict_anchor_orientation: bool,
        predict_mano: bool,
        share_decoder_for_all_tasks: bool,
    ) -> None:
        super().__init__()
        self._share_decoder_for_all_tasks = share_decoder_for_all_tasks
        self.choir_dim = 2  # 0: closest object point distance, 1: fixed anchor distance

        # ======================= Encoder =======================
        encoder: List[torch.nn.Module] = [
            torch.nn.Linear(self.choir_dim * bps_dim, encoder_layer_dims[0]),
            torch.nn.BatchNorm1d(encoder_layer_dims[0]),
            torch.nn.ReLU(),
        ]
        for i in range(len(encoder_layer_dims) - 1):
            encoder.append(
                torch.nn.Linear(encoder_layer_dims[i], encoder_layer_dims[i + 1])
            )
            encoder.append(torch.nn.BatchNorm1d(encoder_layer_dims[i + 1]))
            encoder.append(torch.nn.ReLU())
        encoder.append(torch.nn.Linear(encoder_layer_dims[-1], latent_dim))
        self.encoder = torch.nn.Sequential(*encoder)
        # ========================================================
        # ======================= Decoder =======================
        decoder: List[torch.nn.Module] = [
            torch.nn.Linear(latent_dim, decoder_layer_dims[0]),
            torch.nn.BatchNorm1d(decoder_layer_dims[0]),
            torch.nn.ReLU(),
        ]
        for i in range(len(decoder_layer_dims) - 1):
            decoder.append(
                torch.nn.Linear(decoder_layer_dims[i], decoder_layer_dims[i + 1])
            )
            decoder.append(torch.nn.BatchNorm1d(decoder_layer_dims[i + 1]))
            decoder.append(torch.nn.ReLU())
        self.decoder = torch.nn.Sequential(*decoder)
        self._anchor_dist_decoder = torch.nn.Sequential(
            torch.nn.Linear(decoder_layer_dims[-1], bps_dim),
        )
        # ========================================================
        # ======================= Anchor ========================
        self.anchor_orientation_decoder = None
        if predict_anchor_orientation:
            raise NotImplementedError
            if not share_decoder_for_all_tasks:
                anchor_orientation_decoder: List[torch.nn.Module] = [
                    torch.nn.Linear(latent_dim, decoder_layer_dims[0]),
                    torch.nn.BatchNorm1d(decoder_layer_dims[0]),
                    torch.nn.ReLU(),
                ]
                for i in range(len(decoder_layer_dims) - 1):
                    anchor_orientation_decoder.append(
                        torch.nn.Linear(
                            decoder_layer_dims[i], decoder_layer_dims[i + 1]
                        )
                    )
                    anchor_orientation_decoder.append(
                        torch.nn.BatchNorm1d(decoder_layer_dims[i + 1])
                    )
                    anchor_orientation_decoder.append(torch.nn.ReLU())
                anchor_orientation_decoder.append(
                    torch.nn.Linear(decoder_layer_dims[-1], 3 * bps_dim)
                )
                self.anchor_orientation_decoder = torch.nn.Sequential(
                    *anchor_orientation_decoder
                )
            else:
                self.anchor_orientation_decoder = torch.nn.Linear(
                    decoder_layer_dims[-1], 3 * bps_dim
                )

        # ========================================================
        # ======================= MANO ===========================
        self.mano_decoder, self.mano_params_decoder, self.mano_pose_decoder = (
            None,
            None,
            None,
        )
        if predict_mano:
            if not share_decoder_for_all_tasks:
                mano_decoder: List[torch.nn.Module] = [
                    torch.nn.Linear(latent_dim, decoder_layer_dims[0]),
                    torch.nn.BatchNorm1d(decoder_layer_dims[0]),
                    torch.nn.ReLU(),
                ]
                for i in range(len(decoder_layer_dims) - 1):
                    mano_decoder.append(
                        torch.nn.Linear(
                            decoder_layer_dims[i], decoder_layer_dims[i + 1]
                        )
                    )
                    mano_decoder.append(torch.nn.BatchNorm1d(decoder_layer_dims[i + 1]))
                    mano_decoder.append(torch.nn.ReLU())
                # mano_decoder.append(torch.nn.Linear(decoder_layer_dims[-1], 10 + 18))
                self.mano_decoder = torch.nn.Sequential(*mano_decoder)
            self.mano_params_decoder = (
                torch.nn.Sequential(
                    torch.nn.Linear(decoder_layer_dims[-1], decoder_layer_dims[-1]),
                    torch.nn.BatchNorm1d(decoder_layer_dims[-1]),
                    torch.nn.ReLU(),
                    torch.nn.Linear(
                        decoder_layer_dims[-1], 10 + 18
                    ),  # 10 for shape, 18 for pose
                )
                if share_decoder_for_all_tasks
                else torch.nn.Linear(decoder_layer_dims[-1], 10 + 18)
            )
            self.mano_pose_decoder = (
                torch.nn.Sequential(
                    torch.nn.Linear(decoder_layer_dims[-1], decoder_layer_dims[-1]),
                    torch.nn.BatchNorm1d(decoder_layer_dims[-1]),
                    torch.nn.ReLU(),
                    torch.nn.Linear(
                        decoder_layer_dims[-1], 6 + 3
                    ),  # 6 for rotation, 3 for translation
                )
                if share_decoder_for_all_tasks
                else torch.nn.Linear(decoder_layer_dims[-1], 6 + 3)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape
        B, P, _ = input_shape
        _x = x
        latent = self.encoder(x.flatten(start_dim=1))
        x = self.decoder(latent)

        anchor_orientations, mano = None, None
        if self.anchor_orientation_decoder is not None:
            anchor_orientations = self.anchor_orientation_decoder(
                x if self._share_decoder_for_all_tasks else latent
            ).view(B, P, 3)
            anchor_orientations = torch.nn.functional.normalize(
                anchor_orientations, dim=-1
            )
        if self.mano_params_decoder is not None and self.mano_pose_decoder is not None:
            mano_embedding = (
                x if self._share_decoder_for_all_tasks else self.mano_decoder(latent)
            )
            mano_params = self.mano_params_decoder(mano_embedding)
            mano_pose = self.mano_pose_decoder(mano_embedding)
            mano = torch.cat([mano_params, mano_pose], dim=-1)

        sqrt_distances = self._anchor_dist_decoder(x)
        anchor_dist = (sqrt_distances**2).view(B, P, 1)  # N, 1
        choir = torch.cat(
            [
                _x.view(B, P, self.choir_dim)[:, :, 0]
                .unsqueeze(-1)
                .requires_grad_(False),
                anchor_dist,
            ],
            dim=-1,
        )
        assert choir.shape == input_shape, f"{choir.shape} != {input_shape}"
        return {"choir": choir, "orientations": anchor_orientations, "mano": mano}
