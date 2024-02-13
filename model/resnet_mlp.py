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

from model.attention import MultiHeadAttention


class TemporalResidualBlock(torch.nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        n_norm_groups: int = 32,
        temporal_dim: Optional[int] = None,
        y_dim: Optional[int] = None,
    ):
        super().__init__()
        self.lin1 = torch.nn.Linear(
            dim_in,
            dim_out,
        )
        self.norm1 = torch.nn.GroupNorm(n_norm_groups, dim_out)
        self.nonlin = torch.nn.SiLU()
        self.lin2 = torch.nn.Linear(
            dim_out,
            dim_out,
        )
        self.norm2 = torch.nn.GroupNorm(n_norm_groups, dim_out)
        self.out_activation = torch.nn.SiLU()
        self.temporal_projection = (
            torch.nn.Sequential(
                torch.nn.SiLU(),
                torch.nn.Linear(
                    temporal_dim,
                    dim_out * 2,
                ),
            )
            if temporal_dim is not None
            else None
        )
        self.cross_attention = (
            MultiHeadAttention(
                q_dim=dim_out, k_dim=y_dim, v_dim=y_dim, n_heads=8, dim_head=64
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
        t_emb: Optional[torch.Tensor] = None,
        y_emb: Optional[torch.Tensor] = None,
        debug: bool = False,
    ) -> torch.Tensor:
        def print_debug(str):
            nonlocal debug
            if debug:
                print(str)

        _x = x
        # print_debug(f"Starting with x.shape = {x.shape}")
        if t_emb is not None:
            scale, shift = self.temporal_projection(t_emb).chunk(2, dim=1)
            # print_debug(f"scale and shift shapes: {scale.shape}, {shift.shape}")
        x = self.lin1(x)
        # print_debug(f"After lin1, x.shape = {x.shape}")
        x = self.norm1(x)
        if t_emb is not None:
            x = (
                x * (scale + 1) + shift
            )  # Normalize before scale/shift as in OpenAI's code
        x = self.nonlin(x)
        # if t_emb is not None:
        # print_debug(f"Temb is {t_emb.shape}")
        # print_debug(f"Temb projected is {self.temporal_projection(t_emb).shape}")
        x = self.lin2(x)
        # print_debug(f"After lin2, x.shape = {x.shape}")
        if y_emb is not None:
            assert t_emb is not None
            # print_debug(f"Adding _x of shape {_x.shape} to x of shape {x.shape}")
            x = self.out_activation(self.norm2(x + self.residual_scaling(_x)))
            # print_debug(
            # f"y_emb is {y_emb.shape}. Adding result of cross-attention to x."
            # )
            if len(y_emb.shape) == 2:
                y_emb = y_emb[:, None]
            cross_attn = self.cross_attention(q=x[:, None], k=y_emb, v=y_emb).squeeze(
                dim=1
            )
            # print_debug(f"Cross-attention result is {cross_attn.shape}")
            x = x + cross_attn
            return x
        else:
            pass
            # print_debug(
            # f"Adding _x of shape {_x.shape} (rescaled to {self.residual_scaling(_x).shape}) to x of shape {x.shape}"
            # )
            return self.out_activation(self.norm2(x + self.residual_scaling(_x)))


class ResidualBlock(TemporalResidualBlock):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        n_norm_groups: int = 32,
    ):
        super().__init__(
            dim_in,
            dim_out,
            n_norm_groups=n_norm_groups,
            temporal_dim=None,
            y_dim=None,
        )
