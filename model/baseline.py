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
        anchor_assignment: str,
        encoder_layer_dims: Tuple[int],
        decoder_layer_dims: Tuple[int],
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
        )
        self._anchor_class_decoder = (
            torch.nn.Sequential(
                torch.nn.Linear(
                    decoder_layer_dims[-1],
                    bps_dim * 32 * (1 if self.single_anchor else 2),
                ),
                torch.nn.Softmax(dim=1),
            )
            if anchor_assignment == "closest"
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape
        B, P, _ = input_shape
        _x = x
        x = self.encoder(x.flatten(start_dim=1))
        x = self.decoder(x)
        # bps_dist = self._bps_dist_decoder(x).view(B, P, 1)  # N, 1
        # bps_deltas = self._bps_deltas_decoder(x).view(B, P, 3)  # N, 3
        sqrt_distances = self._anchor_dist_decoder(x)
        anchor_dist = (sqrt_distances**2).view(
            B, P, 1 if self.single_anchor else 2
        )  # N, 1
        if self._anchor_class_decoder is not None:
            anchor_class = self._anchor_class_decoder(x).view(
                B, P, 32 * (1 if self.single_anchor else 2)
            )  # N, 32 (one-hot)
        else:
            if self.single_anchor:
                anchor_class = _x.view(B, P, self.choir_dim)[:, :, -32:].requires_grad_(
                    False
                )
            else:
                raise NotImplementedError
        # choir = torch.cat([bps_dist, bps_deltas, anchor_dist, anchor_class], dim=-1)
        choir = torch.cat(
            [
                _x.view(B, P, self.choir_dim)[:, :, :4].requires_grad_(False),
                anchor_dist,
                anchor_class,
            ],
            dim=-1,
        )
        assert choir.shape == input_shape, f"{choir.shape} != {input_shape}"
        return choir


class BaselineUNetModel(torch.nn.Module):
    def __init__(
        self,
        bps_dim: int,
        anchor_assignment: str,
        encoder_layer_dims: Tuple[int],
        decoder_layer_dims: Tuple[int],
        latent_dim: int,
    ) -> None:
        super().__init__()
        self.single_anchor = anchor_assignment in ["closest", "random"]
        self.choir_dim = 37 if self.single_anchor else 4 + (33 * 2)
        encoder_mlp: List[torch.nn.Module] = [
            torch.nn.Linear(self.choir_dim * bps_dim, encoder_layer_dims[0]),
        ]
        encoder_bn: List[torch.nn.Module] = [
            torch.nn.BatchNorm1d(encoder_layer_dims[0])
        ]
        for i in range(len(encoder_layer_dims) - 1):
            encoder_mlp.append(
                torch.nn.Linear(encoder_layer_dims[i], encoder_layer_dims[i + 1])
            )
            encoder_bn.append(torch.nn.BatchNorm1d(encoder_layer_dims[i + 1]))
        encoder_mlp.append(torch.nn.Linear(encoder_layer_dims[-1], latent_dim))
        self.encoder_mlp = torch.nn.ModuleList(encoder_mlp)
        self.encoder_bn = torch.nn.ModuleList(encoder_bn)
        decoder_mlp: List[torch.nn.Module] = [
            torch.nn.Linear(latent_dim + encoder_layer_dims[-1], decoder_layer_dims[0]),
        ]
        decoder_bn: List[torch.nn.Module] = [
            torch.nn.BatchNorm1d(decoder_layer_dims[0]),
        ]
        for i in range(len(decoder_layer_dims) - 1):
            decoder_mlp.append(
                torch.nn.Linear(
                    decoder_layer_dims[i] + encoder_layer_dims[-2 - i],
                    decoder_layer_dims[i + 1],
                )
            )
            decoder_bn.append(torch.nn.BatchNorm1d(decoder_layer_dims[i + 1]))
        self.decoder_mlp = torch.nn.ModuleList(decoder_mlp)
        self.decoder_bn = torch.nn.ModuleList(decoder_bn)
        # self._bps_dist_decoder = torch.nn.Sequential(
        # torch.nn.Linear(decoder_layer_dims[-1], bps_dim), torch.nn.ReLU()
        # )
        # self._bps_deltas_decoder = torch.nn.Linear(decoder_layer_dims[-1], bps_dim * 3)
        self._anchor_dist_decoder = torch.nn.Sequential(
            torch.nn.Linear(
                decoder_layer_dims[-1], bps_dim * (1 if self.single_anchor else 2)
            ),
        )
        self._anchor_class_decoder = (
            torch.nn.Sequential(
                torch.nn.Linear(
                    decoder_layer_dims[-1],
                    bps_dim * 32 * (1 if self.single_anchor else 2),
                ),
                torch.nn.Softmax(dim=1),
            )
            if anchor_assignment == "closest"
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape
        B, P, _ = input_shape
        _x = x
        x = x.flatten(start_dim=1)
        encoder_outputs = []
        for linear, bn in zip(self.encoder_mlp[:-1], self.encoder_bn):
            x = bn(torch.nn.functional.relu(linear(x)))
            encoder_outputs.append(x)
        x = self.encoder_mlp[-1](x)  # Latent embedding
        for i, (linear, bn) in enumerate(zip(self.decoder_mlp, self.decoder_bn)):
            x = torch.cat([x, encoder_outputs[-1 - i]], dim=-1)
            x = bn(torch.nn.functional.relu(linear(x)))

        # bps_dist = self._bps_dist_decoder(x).view(B, P, 1)  # N, 1
        # bps_deltas = self._bps_deltas_decoder(x).view(B, P, 3)  # N, 3
        sqrt_distances = self._anchor_dist_decoder(x)
        anchor_dist = (sqrt_distances**2).view(
            B, P, 1 if self.single_anchor else 2
        )  # N, 1
        if self._anchor_class_decoder is not None:
            anchor_class = self._anchor_class_decoder(x).view(
                B, P, 32 * (1 if self.single_anchor else 2)
            )  # N, 32 (one-hot)
        else:
            if self.single_anchor:
                anchor_class = _x.view(B, P, self.choir_dim)[:, :, -32:].requires_grad_(
                    False
                )
            else:
                raise NotImplementedError
        # choir = torch.cat([bps_dist, bps_deltas, anchor_dist, anchor_class], dim=-1)
        choir = torch.cat(
            [
                _x.view(B, P, self.choir_dim)[:, :, :4].requires_grad_(False),
                anchor_dist,
                anchor_class,
            ],
            dim=-1,
        )
        assert choir.shape == input_shape, f"{choir.shape} != {input_shape}"
        return choir
