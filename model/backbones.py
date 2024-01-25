#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Backbones for all sorts of things (diffusion hehehehehehehe).
"""

from functools import partial
from typing import Optional, Tuple

import torch

from model.attention import SpatialTransformer
from model.resnet_conv import DownScaleResidualBlock as ConvDownBlock
from model.resnet_conv import IdentityResidualBlock as ConvIdentityBlock
from model.resnet_conv import TemporalDownScaleResidualBlock as TemporalConvDownBlock
from model.resnet_conv import TemporalIdentityResidualBlock as TemporalConvIdentityBlock
from model.resnet_conv import TemporalUpScaleResidualBlock as TemporalConvUpBlock
from model.resnet_mlp import ResidualBlock as ResBlock
from model.resnet_mlp import TemporalResidualBlock as TemporalResBlock
from vendor.Pointnet_Pointnet2_pytorch.models.pointnet2_utils import (
    PointNetFeaturePropagation,
    PointNetSetAbstraction,
)


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
        input_dim: int,
        output_dim: int,
        temporal_dim: int,
        hidden_dim: int = 1024,
        context_dim: Optional[int] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.time_encoder = time_encoder
        self.time_mlp = torch.nn.Sequential(
            torch.nn.Linear(temporal_dim, temporal_dim),
            torch.nn.GELU(),
            torch.nn.Linear(temporal_dim, temporal_dim),
        )
        self.input_layer = torch.nn.Sequential(
            torch.nn.Linear(
                input_dim,
                hidden_dim,
            ),
            torch.nn.GroupNorm(32, hidden_dim),
            torch.nn.SiLU(),
        )
        temporal_res_block = partial(
            TemporalResBlock,
            dim_in=hidden_dim,
            dim_out=hidden_dim,
            n_norm_groups=32,
            temporal_dim=temporal_dim,
            y_dim=context_dim,
        )
        self.block_1 = temporal_res_block()
        self.block_2 = temporal_res_block()
        self.block_3 = temporal_res_block()
        self.block_4 = temporal_res_block()
        self.output_layer = torch.nn.Linear(hidden_dim, output_dim)
        self.output_dim = output_dim

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
        x1 = self.input_layer(x)
        x2 = self.block_1(x1, t_emb=time_embed, y_emb=y, debug=debug)
        x3 = self.block_2(x2, t_emb=time_embed, y_emb=y, debug=debug)
        x4 = self.block_3(x3, t_emb=time_embed, y_emb=y, debug=debug)
        x5 = self.block_4(x4, t_emb=time_embed, y_emb=y, debug=debug)
        # x5 = x4
        x6 = self.output_layer(x5)
        output_shape = (*input_shape[:-2], self.output_dim // 3, 3)
        return x6.view(output_shape)


class PointNet2EncoderModel(torch.nn.Module):
    def __init__(
        self,
        input_points: int,
        embed_dim: int = 128,
    ):
        super().__init__()
        self.sa1 = PointNetSetAbstraction(
            npoint=512,
            radius=0.2,
            nsample=32,
            in_channel=3,
            mlp=[64, 64, 128],
            group_all=False,
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=128,
            radius=0.4,
            nsample=64,
            in_channel=128 + 3,
            mlp=[128, 128, 256],
            group_all=False,
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=256 + 3,
            mlp=[256, 512, 1024],
            group_all=True,
        )
        self.fc1 = torch.nn.Linear(1024, 512)
        self.bn1 = torch.nn.BatchNorm1d(512)
        self.drop1 = torch.nn.Dropout(0.4)
        self.fc2 = torch.nn.Linear(512, 256)
        self.bn2 = torch.nn.BatchNorm1d(256)
        self.drop2 = torch.nn.Dropout(0.4)
        self.fc3 = torch.nn.Linear(256, embed_dim)

    def forward(self, points):
        B, T, N, C = points.shape
        points = points.view(B * T, N, C)
        points = points.permute(0, 2, 1)  # (batch_size, 3, num_points)
        normals = None
        l1_xyz, l1_points = self.sa1(points, normals)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B * T, 1024)
        x = self.drop1(torch.nn.functional.gelu(self.bn1(self.fc1(x))))
        x = self.drop2(torch.nn.functional.gelu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x.view(B, T, -1)


class PointNet2EncoderModel_old(torch.nn.Module):
    def __init__(
        self,
        input_points: int,
        embed_dim: int = 128,
    ):
        super().__init__()
        self.sa1 = PointNetSetAbstraction(512, 0.1, 32, 3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = torch.nn.Conv1d(128, 64, 1)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.drop1 = torch.nn.Dropout(0.5)
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(128 * input_points, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.GELU(),
            torch.nn.Linear(1024, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.GELU(),
            torch.nn.Linear(512, embed_dim),
        )

    def forward(self, points):
        b, t, n, c = points.shape
        points = points.view(b * t, n, c)
        points = points.permute(0, 2, 1)  # (batch_size, 3, num_points)
        l0_points = None
        l0_xyz = points[:, :3, :]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = self.drop1(
            torch.nn.functional.relu(self.bn1(self.conv1(l0_points)))
        )  # (batch_size, c, num_points)
        print(f"Output shape before FF = {x.shape}")
        x = x.view(b, t, -1)
        x = self.ff(x)
        print(f"Output shape after FF = {x.shape}")
        return x


class MLPResNetEncoderModel(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        embed_dim: int = 128,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.input_layer = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 8192),
            torch.nn.GroupNorm(32, 8192),
            torch.nn.SiLU(),
        )
        self.block_1 = ResBlock(8192, hidden_dim)
        self.block_2 = ResBlock(hidden_dim, hidden_dim)
        self.block_3 = ResBlock(hidden_dim, hidden_dim)
        # self.block_4 = ResBlock(hidden_dim, hidden_dim)
        self.output_layer = torch.nn.Linear(hidden_dim, embed_dim)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = x.view(x.shape[0], -1)
        x1 = self.input_layer(x)
        x2 = self.block_1(x1)
        x3 = self.block_2(x2)
        # x4 = x3
        x4 = self.block_3(x3)
        # x5 = self.block_4(x4)
        x5 = x4
        x6 = self.output_layer(x5)
        return x6


class UNetBackboneModel(torch.nn.Module):
    def __init__(
        self,
        time_encoder: torch.nn.Module,
        bps_grid_len: int,
        choir_dim: int,
        output_channels: int,
        temporal_dim: int,
        pooling: str = "avg",
        normalization: str = "batch",
        norm_groups: int = 16,
        output_paddings: Tuple[int] = (1, 1, 1, 1),
        context_channels: Optional[int] = None,
        use_self_attention: bool = False,
    ):
        super().__init__()
        self.grid_len = bps_grid_len
        self.choir_dim = choir_dim
        self.time_encoder = time_encoder
        self.use_self_attn = use_self_attention
        self.time_mlp = torch.nn.Sequential(
            torch.nn.Linear(temporal_dim, temporal_dim),
            torch.nn.GELU(),
            torch.nn.Linear(temporal_dim, temporal_dim),
        )
        # ========= Partials =========
        down_conv_block = partial(
            TemporalConvDownBlock,
            temporal_channels=temporal_dim,
            normalization=normalization,
            norm_groups=norm_groups,
            context_channels=context_channels,
            pooling=pooling,
        )
        up_conv_block = partial(
            TemporalConvUpBlock,
            temporal_channels=temporal_dim,
            normalization=normalization,
            norm_groups=norm_groups,
            context_channels=context_channels,
            interpolate=False,
        )
        identity_conv_block = partial(
            TemporalConvIdentityBlock,
            temporal_channels=temporal_dim,
            normalization=normalization,
            norm_groups=norm_groups,
            context_channels=context_channels,
        )
        spatial_transformer = partial(
            SpatialTransformer,
            n_heads=8,
            dim_heads=32,
            dropout=0.0,
            gated_ff=True,
            norm_groups=norm_groups,
        )
        # ========= Layers =========
        dim_heads = 32
        self.identity1 = identity_conv_block(
            channels_in=self.choir_dim,
            channels_out=64,
            norm_groups=min(16, norm_groups),
        )
        self.down1 = down_conv_block(
            channels_in=64, channels_out=64, norm_groups=min(16, norm_groups)
        )
        # self.self_attn_1 = (
        # spatial_transformer(
        # in_channels=64, n_heads=64 // dim_heads, dim_heads=dim_heads
        # )
        # if use_self_attention
        # else None
        # )
        self.down2 = down_conv_block(channels_in=64, channels_out=128)
        self.self_attn_2 = (
            spatial_transformer(
                in_channels=128, n_heads=128 // dim_heads, dim_heads=dim_heads
            )
            if use_self_attention
            else None
        )
        self.down3 = down_conv_block(channels_in=128, channels_out=256)
        self.self_attn_3 = (
            spatial_transformer(
                in_channels=256, n_heads=256 // dim_heads, dim_heads=dim_heads
            )
            if use_self_attention
            else None
        )
        self.tunnel1 = identity_conv_block(channels_in=256, channels_out=256)
        self.self_attn_4 = (
            spatial_transformer(
                in_channels=256, n_heads=256 // dim_heads, dim_heads=dim_heads
            )
            if use_self_attention
            else None
        )
        self.tunnel2 = identity_conv_block(channels_in=256, channels_out=256)
        self.up1 = up_conv_block(
            channels_in=512, channels_out=128, output_padding=output_paddings[0]
        )
        self.self_attn_5 = (
            spatial_transformer(
                in_channels=128, n_heads=128 // dim_heads, dim_heads=dim_heads
            )
            if use_self_attention
            else None
        )
        self.up2 = up_conv_block(
            channels_in=256, channels_out=64, output_padding=output_paddings[1]
        )
        self.self_attn_6 = (
            spatial_transformer(
                in_channels=64, n_heads=64 // dim_heads, dim_heads=dim_heads
            )
            if use_self_attention
            else None
        )
        self.up3 = up_conv_block(
            channels_in=128, channels_out=64, output_padding=output_paddings[2]
        )
        # self.self_attn_7 = (
        # spatial_transformer(
        # in_channels=64, n_heads=64 // dim_heads, dim_heads=dim_heads
        # )
        # if use_self_attention
        # else None
        # )
        self.identity3 = identity_conv_block(
            channels_in=64, channels_out=64, norm_groups=min(16, norm_groups)
        )
        self.out_conv = torch.nn.Conv3d(64, output_channels, 1, padding=0, stride=1)
        self.output_channels = output_channels

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.tensor] = None,
        debug: bool = False,
    ) -> torch.Tensor:
        input_shape = x.shape
        bs, ctx_len = x.shape[0], (x.shape[1] if len(x.shape) == 4 else 1)
        if ctx_len > 1:
            # Repeat the context N times if generating N samples.
            y = (
                y[:, None, ...]
                .repeat(1, ctx_len, 1, 1, 1, 1)
                .view(1 * ctx_len, *y.shape[1:])
            )
        comp_shape = (
            bs * ctx_len,
            self.grid_len,
            self.grid_len,
            self.grid_len,
            self.choir_dim,
        )
        t = t.view(bs * ctx_len, -1)
        x = x.view(*comp_shape).permute(0, 4, 1, 2, 3)
        assert x.shape[0] == t.shape[0], "Batch size mismatch between x and t"
        assert x.shape[0] == y.shape[0], "Batch size mismatch between x and y"
        t_embed = self.time_mlp(self.time_encoder(t))
        x1 = self.identity1(x, t_embed, context=y, debug=debug)
        x2 = self.down1(x1, t_embed, context=y, debug=debug)
        # x2 = self.self_attn_1(x2) if self.use_self_attn else x2
        x3 = self.down2(x2, t_embed, context=y, debug=debug)
        x3 = self.self_attn_2(x3) if self.use_self_attn else x3
        x4 = self.down3(x3, t_embed, context=y, debug=debug)
        x4 = self.self_attn_3(x4) if self.use_self_attn else x4
        x5 = self.tunnel1(x4, t_embed, context=y, debug=debug)
        x5 = self.self_attn_4(x5) if self.use_self_attn else x5
        x6 = self.tunnel2(x5, t_embed, context=y, debug=debug)
        # x6 = self.cross_attn5(x6, context=y) if self.use_spatial_transformer else x6
        # The output of the final downsampling layer is concatenated with the output of the final
        # tunnel layer because they have the same shape H and W. Then we upscale those features and
        # conctenate the upscaled features with the output of the previous downsampling layer, and
        # so on.
        x7 = self.up1(torch.cat((x6, x4), dim=1), t_embed, context=y, debug=debug)
        x7 = self.self_attn_5(x7) if self.use_self_attn else x7
        x8 = self.up2(torch.cat((x7, x3), dim=1), t_embed, context=y, debug=debug)
        x8 = self.self_attn_6(x8) if self.use_self_attn else x8
        x9 = self.up3(torch.cat((x8, x2), dim=1), t_embed, context=y, debug=debug)
        # x9 = self.self_attn_7(x9) if self.use_self_attn else x9
        x10 = self.identity3(x9, t_embed, context=y, debug=debug)
        comp_output_shape = (
            bs * ctx_len,
            self.output_channels,
            self.grid_len,
            self.grid_len,
            self.grid_len,
        )
        output_shape = (
            *input_shape[:-1],
            self.output_channels,
        )
        x = (
            self.out_conv(x10)
            .permute(0, 2, 3, 4, 1)
            .reshape(comp_output_shape)
            .view(output_shape)
        )
        return x


class ResnetEncoderModel(torch.nn.Module):
    def __init__(
        self,
        bps_grid_len: int,
        choir_dim: int,
        embed_channels: int,
        pooling: str = "max",
        normalization: str = "batch",
        norm_groups: int = 16,
        pool_all_features: str = "none",
        use_self_attention: bool = False,
    ):
        super().__init__()
        assert pool_all_features in ["none", "spatial", "attention", "adaptive"]
        self.grid_len = bps_grid_len
        self.choir_dim = choir_dim
        self.use_self_attn = use_self_attention
        spatial_transformer = partial(
            SpatialTransformer,
            n_heads=8,
            dim_heads=32,
            dropout=0.0,
            gated_ff=False,
            norm_groups=norm_groups,
        )
        dim_heads = 32
        # self.identity = ConvIdentityBlock(
        # choir_dim,
        # 32,
        # normalization=normalization,
        # norm_groups=norm_groups,
        # )
        self.down1 = ConvDownBlock(
            choir_dim,
            64,
            normalization=normalization,
            norm_groups=norm_groups,
            pooling=pooling,
        )
        self.self_attn_1 = (
            spatial_transformer(
                in_channels=64, n_heads=64 // dim_heads, dim_heads=dim_heads
            )
            if use_self_attention
            else None
        )
        self.down2 = ConvDownBlock(
            64,
            128,
            normalization=normalization,
            norm_groups=norm_groups,
            pooling=pooling,
        )
        self.down3 = ConvDownBlock(
            128,
            256,
            normalization=normalization,
            norm_groups=norm_groups,
            pooling=pooling,
        )
        # self.self_attn_3 = (
        #    spatial_transformer(
        #        in_channels=256, n_heads=256 // dim_heads, dim_heads=dim_heads
        #    )
        #    if use_self_attention
        #    else None
        # )
        self.out_identity = ConvIdentityBlock(
            256,
            embed_channels,
            normalization=normalization,
            norm_groups=norm_groups,
        )
        self.self_attn_out = (
            spatial_transformer(
                in_channels=embed_channels,
                n_heads=embed_channels // dim_heads,
                dim_heads=dim_heads,
            )
            if use_self_attention
            else None
        )
        self.out_conv = (
            torch.nn.Conv3d(embed_channels, embed_channels, 1, padding=0, stride=1)
            if pool_all_features == "none"
            else None
        )
        self.pooling = None
        self.pooling_method = pool_all_features
        feature_dim = 64 + 128 + 256 + embed_channels
        if pool_all_features == "spatial":
            self.pooling = torch.nn.Sequential(
                torch.nn.Linear(feature_dim, 1024),
                torch.nn.GELU(),
                torch.nn.Linear(1024, embed_channels),
            )
        elif pool_all_features == "adaptive":
            self.pooling = torch.nn.Sequential(
                torch.nn.AdaptiveAvgPool3d(1),
                torch.nn.Conv3d(embed_channels, embed_channels, 1),
            )

    def forward(
        self,
        x: torch.Tensor,
        debug: bool = False,
    ) -> torch.Tensor:
        if len(x.shape) == 3:
            bs, n, ctx_len = x.shape[0], 1, 1
        elif len(x.shape) == 4:
            bs, n, ctx_len = x.shape[0], 1, x.shape[1]
        elif len(x.shape) == 5:
            bs, n, ctx_len = x.shape[0], x.shape[1], x.shape[2]
        interm_features = []
        x = x.view(
            bs * n * ctx_len,
            self.grid_len,
            self.grid_len,
            self.grid_len,
            self.choir_dim,
        ).permute(0, 4, 1, 2, 3)
        # x = self.identity(x, debug=debug)
        # if self.pooling_method.startswith("spatial"):
        # interm_features.append(x.mean(dim=(2, 3, 4)))
        x = self.down1(x, debug=debug)
        if self.use_self_attn:
            x = self.self_attn_1(x)
        if self.pooling_method.startswith("spatial"):
            interm_features.append(x.mean(dim=(2, 3, 4)))
        x = self.down2(x, debug=debug)
        if self.pooling_method.startswith("spatial"):
            interm_features.append(x.mean(dim=(2, 3, 4)))
        x = self.down3(x, debug=debug)
        # if self.use_self_attn:
        # x = self.self_attn_3(x)
        if self.pooling_method.startswith("spatial"):
            interm_features.append(x.mean(dim=(2, 3, 4)))
        x = self.out_identity(x, debug=debug)
        if self.use_self_attn:
            x = self.self_attn_out(x)
        if self.pooling_method.startswith("spatial"):
            interm_features.append(x.mean(dim=(2, 3, 4)))

        if self.pooling_method == "none":
            x = self.out_conv(x)
            x = x.view(
                bs * n, ctx_len, -1, x.shape[-3], x.shape[-2], x.shape[-1]
            ).squeeze(
                dim=1
            )  # For now we'll try with 1 context frame TODO
        elif self.pooling_method.startswith("spatial"):
            x = torch.cat(interm_features, dim=-1)
            x = self.pooling(x)
            x = x.view(bs * n, ctx_len, -1, 1, 1, 1).squeeze(
                dim=1
            )  # For now we'll try with 1 context frame TODO
        else:
            x = self.pooling(x)
            x = x.view(
                bs * n, ctx_len, -1, x.shape[-3], x.shape[-2], x.shape[-1]
            ).squeeze(
                dim=1
            )  # For now we'll try with 1 context frame TODO
        return x
