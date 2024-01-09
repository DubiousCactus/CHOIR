#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
1D ResNet building blocks using fully connected layers.
"""


from typing import Optional

import torch

from model.attention import VectorAttentionLayer


class ResidualBlock(torch.nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
    ):
        super().__init__()
        self.lin1 = torch.nn.Linear(
            dim_in,
            dim_out,
        )
        self.norm1 = torch.nn.LayerNorm(dim_out)
        self.nonlin = torch.nn.SiLU()
        self.lin2 = torch.nn.Linear(
            dim_out,
            dim_out,
        )
        self.norm2 = torch.nn.LayerNorm(dim_out)
        self.out_activation = torch.nn.SiLU()
        self.residual_scaling = (
            torch.nn.Linear(
                dim_in,
                dim_out,
                bias=False,
            )
            if (dim_in != dim_out)
            else torch.nn.Identity()
        )

    def forward(self, x: torch.Tensor, debug: bool = False) -> torch.Tensor:
        def print_debug(str):
            nonlocal debug
            if debug:
                print(str)

        _x = x
        print_debug(f"Starting with x.shape = {x.shape}")
        x = self.lin1(x)
        print_debug(f"After lin1, x.shape = {x.shape}")
        x = self.norm1(x)
        x = self.nonlin(x)
        x = self.lin2(x)
        print_debug(f"After lin2, x.shape = {x.shape}")
        x = self.norm2(x)
        print_debug(
            f"Adding _x of shape {_x.shape} (rescaled to {self.residual_scaling(_x).shape}) to x of shape {x.shape}"
        )
        return self.out_activation(x + self.residual_scaling(_x))


class TemporalResidualBlock(torch.nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        temporal_dim: int,
        n_norm_groups: int = 32,
        y_dim: Optional[int] = None,
    ):
        super().__init__()
        self.lin1 = torch.nn.Linear(
            dim_in + (y_dim if y_dim is not None else 0),
            dim_out,
        )
        self.norm1 = torch.nn.GroupNorm(n_norm_groups, dim_out)
        self.nonlin = torch.nn.SiLU()
        self.lin2 = torch.nn.Linear(
            dim_out + (y_dim if y_dim is not None else 0),
            dim_out,
        )
        self.norm2 = torch.nn.GroupNorm(n_norm_groups, dim_out)
        self.out_activation = torch.nn.SiLU()
        self.temporal_projection = torch.nn.Linear(
            temporal_dim,
            dim_out * 2,
        )
        self.cross_attention = (
            VectorAttentionLayer(
                q_dim=dim_out, k_dim=y_dim, v_dim=y_dim, output_dim=y_dim
            )
            if y_dim is not None
            else None
        )
        self.residual_scaling = (
            torch.nn.Linear(
                dim_in,
                dim_out,
                bias=False,
            )
            if (dim_in != dim_out)
            else torch.nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        y_emb: Optional[torch.Tensor] = None,
        debug: bool = False,
    ) -> torch.Tensor:
        def print_debug(str):
            nonlocal debug
            if debug:
                print(str)

        _x = x
        scale, shift = self.temporal_projection(t_emb).chunk(2, dim=1)
        print_debug(f"Starting with x.shape = {x.shape}")
        print_debug(f"scale and shift shapes: {scale.shape}, {shift.shape}")
        x = (
            torch.cat((x, self.cross_attention(q=x, k=y_emb, v=y_emb)), dim=1)
            if y_emb is not None
            else x
        )
        # x = torch.cat((x, y_emb), dim=1) if y_emb is not None else x
        x = self.lin1(x)
        print_debug(f"After lin1, x.shape = {x.shape}")
        x = self.norm1(x)
        x = x * (scale + 1) + shift  # Normalize before scale/shift as in OpenAI's code
        x = self.nonlin(x)
        print_debug(f"Temb is {t_emb.shape}")
        print_debug(f"Temb projected is {self.temporal_projection(t_emb).shape}")
        x = (
            torch.cat((x, self.cross_attention(q=x, k=y_emb, v=y_emb)), dim=1)
            if y_emb is not None
            else x
        )
        # x = torch.cat((x, y_emb), dim=1) if y_emb is not None else x
        x = self.lin2(x)
        print_debug(f"After lin2, x.shape = {x.shape}")
        x = self.norm2(x)
        x = x * (scale + 1) + shift  # Normalize before scale/shift as in OpenAI's code
        print_debug(
            f"Adding _x of shape {_x.shape} (rescaled to {self.residual_scaling(_x).shape}) to x of shape {x.shape}"
        )
        return self.out_activation(x + self.residual_scaling(_x))
