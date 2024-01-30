#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Different positional encoders.
"""

import torch


class SinusoidalPosEncoder(torch.nn.Module):
    def __init__(self, max_positions: int, model_dim: int) -> None:
        super().__init__()
        self.positions = max_positions
        # Little trick to simplify computation:
        constants = torch.exp(
            -torch.arange(0, model_dim, 2)
            * (torch.log(torch.tensor(10000.0)) / model_dim)
        )
        self.pos_embeddings = torch.nn.Parameter(
            torch.zeros(max_positions, model_dim), requires_grad=False
        )
        self.pos_embeddings[:, ::2] = torch.sin(
            torch.arange(0, max_positions).unsqueeze(1).repeat(1, model_dim // 2)
            * constants
        )
        self.pos_embeddings[:, 1::2] = torch.cos(
            torch.arange(0, max_positions).unsqueeze(1).repeat(1, model_dim // 2)
            * constants
        )

    def forward(self, pos: torch.Tensor, flatten_output: bool = True) -> torch.Tensor:
        assert type(pos) == torch.Tensor
        assert len(pos.shape) == 2, "pos must be (B, T)"
        emb = self.pos_embeddings[pos]
        if flatten_output:
            emb = emb.view(pos.shape[0], -1)
        return emb
