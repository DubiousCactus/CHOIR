#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Backbones for all sorts of things (diffusion hehehehehehehe).
"""

import torch

from model.resnet_1d import TemporalResidualBlock as ResBlock


class MLPUNetBackbone(torch.nn.Module):
    def __init__(self, time_encoder: torch.nn.Module, bps_dim: int, temporal_dim: int):
        super().__init__()
        self.choir_dim = 2
        self.time_encoder = time_encoder
        self.time_embedder = torch.nn.Sequential(
            torch.nn.Linear(temporal_dim, temporal_dim),
            torch.nn.GELU(),
            torch.nn.Linear(temporal_dim, temporal_dim),
        )
        self.in_ = torch.nn.Linear(bps_dim * self.choir_dim, 1024)
        self.identity_1 = ResBlock(1024, 1024, temporal_dim)
        self.down_1 = ResBlock(1024, 512, temporal_dim)
        self.down_2 = ResBlock(512, 256, temporal_dim)
        self.down_3 = ResBlock(256, 128, temporal_dim)
        self.tunnel_1 = ResBlock(128, 128, temporal_dim)
        self.tunnel_2 = ResBlock(128, 128, temporal_dim)
        self.up_1 = ResBlock(128 + 128, 256, temporal_dim)
        self.up_2 = ResBlock(256 + 256, 512, temporal_dim)
        self.up_3 = ResBlock(512 + 512, 1024, temporal_dim)
        self.identity_2 = ResBlock(1024, 1024, temporal_dim)
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
        temporal_dim: int,
        hidden_dim: int = 2048,
    ):
        super().__init__()
        self.choir_dim = 1
        self.time_encoder = time_encoder
        self.time_mlp = torch.nn.Sequential(
            torch.nn.Linear(temporal_dim, temporal_dim),
            torch.nn.GELU(),
            torch.nn.Linear(temporal_dim, temporal_dim),
        )
        self.input_layer = torch.nn.Linear(bps_dim * self.choir_dim, hidden_dim)
        self.block_1 = ResBlock(hidden_dim, hidden_dim, temporal_dim)
        self.block_2 = ResBlock(hidden_dim, hidden_dim, temporal_dim)
        self.block_3 = ResBlock(hidden_dim, hidden_dim, temporal_dim)
        self.block_4 = ResBlock(hidden_dim, hidden_dim, temporal_dim)
        self.output_layer = torch.nn.Linear(hidden_dim, bps_dim * self.choir_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape
        x = x.view(x.shape[0], -1)
        time_embed = self.time_mlp(self.time_encoder(t))
        x1 = self.input_layer(x)
        x2 = self.block_1(x1, time_embed)
        x3 = self.block_2(x2, time_embed)
        x4 = self.block_3(x3, time_embed)
        x5 = self.block_4(x4, time_embed)
        return self.output_layer(x5).view(input_shape)
