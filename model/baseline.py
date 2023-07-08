#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Simple baseline to test the data loading pipeline.
"""

from typing import List

import torch


class BaselineModel(torch.nn.Module):
    def __init__(
        self,
        bps_dim: int,
        anchor_assignment: str,
        encoder_layer_dims: List[int],
        decoder_layer_dims: List[int],
        latent_dim: int,
    ) -> None:
        super().__init__()
        self.single_anchor = anchor_assignment in ["closest", "random"]
        self.choir_dim = 37 if self.single_anchor else 4 + (33 * 2)
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
        # self._bps_dist_decoder = torch.nn.Sequential(
        # torch.nn.Linear(decoder_layer_dims[-1], bps_dim), torch.nn.ReLU()
        # )
        # self._bps_deltas_decoder = torch.nn.Linear(decoder_layer_dims[-1], bps_dim * 3)
        self._anchor_dist_decoder = torch.nn.Sequential(
            torch.nn.Linear(
                decoder_layer_dims[-1], bps_dim * (1 if self.single_anchor else 2)
            ),
            torch.nn.ReLU(),
        )
        self._anchor_class_decoder = torch.nn.Sequential(
            torch.nn.Linear(
                decoder_layer_dims[-1], bps_dim * 32 * (1 if self.single_anchor else 2)
            ),
            torch.nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape
        B, P, _ = input_shape
        _x = x
        x = self.encoder(x.flatten(start_dim=1))
        x = self.decoder(x)
        # bps_dist = self._bps_dist_decoder(x).view(B, P, 1)  # N, 1
        # bps_deltas = self._bps_deltas_decoder(x).view(B, P, 3)  # N, 3
        anchor_dist = self._anchor_dist_decoder(x).view(
            B, P, 1 if self.single_anchor else 2
        )  # N, 1
        anchor_class = self._anchor_class_decoder(x).view(
            B, P, 32 * (1 if self.single_anchor else 2)
        )  # N, 32 (one-hot)
        # choir = torch.cat([bps_dist, bps_deltas, anchor_dist, anchor_class], dim=-1)
        choir = torch.cat(
            [_x.view(B, P, self.choir_dim)[:, :, :4], anchor_dist, anchor_class], dim=-1
        )
        assert choir.shape == input_shape, f"{choir.shape} != {input_shape}"
        return choir
