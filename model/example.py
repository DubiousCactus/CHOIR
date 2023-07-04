#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Example model.
"""


import torch


class ExampleModel(torch.nn.Module):
    def __init__(
        self,
        encoder_input_dim: int,
        encoder_dim: int,
        latent_dim: int,
        decoder_output_dim: int,
        decoder_dim: int,
    ) -> None:
        super().__init__()
        self._encoder = torch.nn.Sequential(
            torch.nn.Linear(encoder_input_dim, encoder_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(encoder_dim, latent_dim),
        )
        self._decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, decoder_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(decoder_dim, decoder_output_dim),
        )

    def forward(self, x):
        x = self._encoder(x.view(x.size(0), -1))
        x = self._decoder(x)
        return x
