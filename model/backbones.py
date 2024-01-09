#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Backbones for all sorts of things (diffusion hehehehehehehe).
"""

from typing import Optional, Tuple

import torch

from model.resnet_conv import TemporalDownScaleResidualBlock as TemporalConvDownBlock
from model.resnet_conv import TemporalIdentityResidualBlock as TemporalConvIdentityBlock
from model.resnet_conv import TemporalUpScaleResidualBlock as TemporalConvUpBlock
from model.resnet_mlp import TemporalResidualBlock as TemporalResBlock


class MLPUNetBackbone(torch.nn.Module):
    def __init__(self, time_encoder: torch.nn.Module, bps_dim: int, temporal_dim: int):
        super().__init__()
        self.choir_dim = 2
        self.time_encoder = time_encoder
        self.time_embedder = torch.nn.Sequential(
            torch.nn.Linear(temporal_dim, temporal_dim),
            torch.nn.GELU(),
            torch.nn.BatchNorm1d(temporal_dim),
            torch.nn.Linear(temporal_dim, temporal_dim),
        )
        self.in_ = torch.nn.Linear(bps_dim * self.choir_dim, 1024)
        self.identity_1 = TemporalResBlock(1024, 1024, temporal_dim)
        self.down_1 = TemporalResBlock(1024, 512, temporal_dim)
        self.down_2 = TemporalResBlock(512, 256, temporal_dim)
        self.down_3 = TemporalResBlock(256, 128, temporal_dim)
        self.tunnel_1 = TemporalResBlock(128, 128, temporal_dim)
        self.tunnel_2 = TemporalResBlock(128, 128, temporal_dim)
        self.up_1 = TemporalResBlock(128 + 128, 256, temporal_dim)
        self.up_2 = TemporalResBlock(256 + 256, 512, temporal_dim)
        self.up_3 = TemporalResBlock(512 + 512, 1024, temporal_dim)
        self.identity_2 = TemporalResBlock(1024, 1024, temporal_dim)
        self.out = torch.nn.Linear(1024, bps_dim * self.choir_dim, temporal_dim)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, debug: bool = False
    ) -> torch.Tensor:
        in_shape = x.shape
        t_emb = self.time_embedder(self.time_encoder(t))
        x1 = self.in_(x.view(in_shape[0], -1))
        x2 = self.identity_1(x1, t_emb)
        x3 = self.down_1(x2, t_emb)
        x4 = self.down_2(x3, t_emb)
        x5 = self.down_3(x4, t_emb)
        x6 = self.tunnel_1(x5, t_emb)
        x7 = self.tunnel_2(x6, t_emb)
        x8 = self.up_1(torch.cat([x7, x5], dim=1), t_emb)
        x9 = self.up_2(torch.cat([x8, x4], dim=1), t_emb)
        x10 = self.up_3(torch.cat([x9, x3], dim=1), t_emb)
        x11 = self.identity_2(x10, t_emb)
        return self.out(x11).view(in_shape)


class MLPResNetBackboneModel(torch.nn.Module):
    def __init__(
        self,
        time_encoder: torch.nn.Module,
        bps_dim: int,
        choir_dim: int,
        temporal_dim: int,
        hidden_dim: int = 2048,
        y_dim: Optional[int] = None,
    ):
        super().__init__()
        self.choir_dim = choir_dim
        self.time_encoder = time_encoder
        self.time_mlp = torch.nn.Sequential(
            torch.nn.Linear(temporal_dim, temporal_dim),
            torch.nn.GELU(),
            torch.nn.BatchNorm1d(temporal_dim),
            torch.nn.Linear(temporal_dim, temporal_dim),
        )
        self.input_layer = torch.nn.Linear(
            bps_dim * self.choir_dim + (y_dim if y_dim is not None else 0), hidden_dim
        )
        self.block_1 = TemporalResBlock(
            hidden_dim, hidden_dim, temporal_dim, y_dim=y_dim
        )
        self.block_2 = TemporalResBlock(
            hidden_dim, hidden_dim, temporal_dim, y_dim=y_dim
        )
        self.block_3 = TemporalResBlock(
            hidden_dim, hidden_dim, temporal_dim, y_dim=y_dim
        )
        self.block_4 = TemporalResBlock(
            hidden_dim, hidden_dim, temporal_dim, y_dim=y_dim
        )
        self.output_layer = torch.nn.Linear(hidden_dim, bps_dim * self.choir_dim)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        debug=False,
    ) -> torch.Tensor:
        input_shape = x.shape
        x = x.view(x.shape[0], -1)
        time_embed = self.time_mlp(self.time_encoder(t))
        x1 = self.input_layer(torch.cat((x, y), dim=-1) if y is not None else x)
        x2 = self.block_1(x1, time_embed, y, debug=debug)
        x3 = self.block_2(x2, time_embed, y, debug=debug)
        x4 = self.block_3(x3, time_embed, y, debug=debug)
        x5 = self.block_4(x4, time_embed, y, debug=debug)
        return self.output_layer(x5).view(input_shape)


class UNetBackboneModel(torch.nn.Module):
    def __init__(
        self,
        time_encoder: torch.nn.Module,
        bps_grid_len: int,
        choir_dim: int,
        temporal_dim: int,
        normalization: str = "batch",
        output_paddings: Tuple[int] = (1, 1, 1, 1),
        y_dim: Optional[int] = None,
    ):
        super().__init__()
        self.grid_len = bps_grid_len
        self.choir_dim = choir_dim
        self.time_encoder = time_encoder
        self.time_mlp = torch.nn.Sequential(
            torch.nn.Linear(temporal_dim, temporal_dim),
            torch.nn.GELU(),
            torch.nn.BatchNorm1d(temporal_dim),
            torch.nn.Linear(temporal_dim, temporal_dim),
        )
        self.identity1 = TemporalConvIdentityBlock(
            self.choir_dim,
            128,
            temporal_dim,
            normalization=normalization,
            conv="3d",
            y_dim=y_dim,
        )
        self.down1 = TemporalConvDownBlock(
            128, 128, temporal_dim, normalization=normalization, conv="3d", y_dim=y_dim
        )
        self.down2 = TemporalConvDownBlock(
            128, 256, temporal_dim, normalization=normalization, conv="3d", y_dim=y_dim
        )
        self.down3 = TemporalConvDownBlock(
            256, 512, temporal_dim, normalization=normalization, conv="3d", y_dim=y_dim
        )
        self.tunnel1 = TemporalConvIdentityBlock(
            512, 512, temporal_dim, normalization=normalization, conv="3d", y_dim=y_dim
        )
        self.tunnel2 = TemporalConvIdentityBlock(
            512, 512, temporal_dim, normalization=normalization, conv="3d", y_dim=y_dim
        )
        self.up1 = TemporalConvUpBlock(
            1024,
            256,
            temporal_dim,
            output_padding=output_paddings[0],
            normalization=normalization,
            conv="3d",
            y_dim=y_dim,
        )
        self.up2 = TemporalConvUpBlock(
            512,
            128,
            temporal_dim,
            output_padding=output_paddings[1],
            normalization=normalization,
            conv="3d",
            y_dim=y_dim,
        )
        self.up3 = TemporalConvUpBlock(
            256,
            64,
            temporal_dim,
            output_padding=output_paddings[2],
            normalization=normalization,
            conv="3d",
            y_dim=y_dim,
        )
        self.identity3 = TemporalConvIdentityBlock(
            64,
            32,
            temporal_dim,
            norm_groups=4,
            normalization=normalization,
            conv="3d",
            y_dim=y_dim,
        )
        self.out_conv = torch.nn.Conv3d(32, self.choir_dim, 1, padding=0, stride=1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.tensor] = None,
        debug: bool = False,
    ) -> torch.Tensor:
        if y is not None:
            raise NotImplementedError("y is not implemented for UNetBackboneModel")
        input_shape = x.shape
        x = x.view(
            x.shape[0], self.grid_len, self.grid_len, self.grid_len, self.choir_dim
        ).permute(0, 4, 1, 2, 3)
        t_embed = self.time_mlp(self.time_encoder(t))
        x1 = self.identity1(x, t_embed, debug=debug)
        x2 = self.down1(x1, t_embed, debug=debug)
        x3 = self.down2(x2, t_embed, debug=debug)
        x4 = self.down3(x3, t_embed, debug=debug)
        x5 = self.tunnel1(x4, t_embed, debug=debug)
        x6 = self.tunnel2(x5, t_embed, debug=debug)
        # The output of the final downsampling layer is concatenated with the output of the final
        # tunnel layer because they have the same shape H and W. Then we upscale those features and
        # conctenate the upscaled features with the output of the previous downsampling layer, and
        # so on.
        x7 = self.up1(torch.cat((x6, x4), dim=1), t_embed, debug=debug)
        x8 = self.up2(torch.cat((x7, x3), dim=1), t_embed, debug=debug)
        x10 = self.up3(torch.cat((x8, x2), dim=1), t_embed, debug=debug)
        x11 = self.identity3(x10, t_embed, debug=debug)
        return self.out_conv(x11).permute(0, 2, 3, 4, 1).reshape(input_shape)


class ConvObjectEncoderModel(torch.nn.Module):
    def __init__(
        self,
        bps_grid_len: int,
        choir_dim: int,
        normalization: str = "batch",
        output_paddings: Tuple[int] = (1, 1, 1, 1),
    ):
        super().__init__()
        self.grid_len = bps_grid_len
        self.choir_dim = choir_dim
        # self.down1 = TemporalConvDownBlock(
        # 64, 64, temporal_dim, normalization=normalization, conv="3d"
        # )
        # self.down2 = TemporalConvDownBlock(
        # 64, 128, temporal_dim, normalization=normalization, conv="3d"
        # )
        # self.down3 = TemporalConvDownBlock(
        # 128, 256, temporal_dim, normalization=normalization, conv="3d"
        # )
        self.out_conv = torch.nn.Conv3d(16, self.choir_dim, 1, padding=0, stride=1)

    def forward(
        self,
        x: torch.Tensor,
        debug: bool = False,
    ) -> torch.Tensor:
        input_shape = x.shape
        x = x.view(
            x.shape[0], self.grid_len, self.grid_len, self.grid_len, self.choir_dim
        ).permute(0, 4, 1, 2, 3)
        return self.out_conv(x).permute(0, 4, 1, 2, 3).reshape(input_shape)
