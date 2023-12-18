#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Different positional time encoders.
"""

import torch


class SinusoidalTimeEncoder(torch.nn.Module):
    def __init__(self, time_steps: int, model_dim: int) -> None:
        super().__init__()
        self.time_steps = time_steps
        # Little trick to simplify computation:
        constants = torch.exp(
            -torch.arange(0, model_dim, 2)
            * (torch.log(torch.tensor(10000.0)) / model_dim)
        )
        self.time_embeddings = torch.nn.Parameter(
            torch.zeros(time_steps, model_dim), requires_grad=False
        )
        self.time_embeddings[:, ::2] = torch.sin(
            torch.arange(0, time_steps).unsqueeze(1).repeat(1, model_dim // 2)
            * constants
        )
        self.time_embeddings[:, 1::2] = torch.cos(
            torch.arange(0, time_steps).unsqueeze(1).repeat(1, model_dim // 2)
            * constants
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        assert type(t) == torch.Tensor
        assert len(t.shape) == 2, "t must be (B, 1)"
        return self.time_embeddings[t].view(t.shape[0], -1)
