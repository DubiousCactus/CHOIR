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
from model.pos_encoding import SinusoidalPosEncoder
from model.resnet_conv import DownScaleResidualBlock as ConvDownBlock
from model.resnet_conv import IdentityResidualBlock as ConvIdentityBlock
from model.resnet_conv import TemporalDownScaleResidualBlock as TemporalConvDownBlock
from model.resnet_conv import TemporalIdentityResidualBlock as TemporalConvIdentityBlock
from model.resnet_conv import TemporalUpScaleResidualBlock as TemporalConvUpBlock
from model.resnet_mlp import ResidualBlock as ResBlock
from model.resnet_mlp import TemporalResidualBlock as TemporalResBlock
from utils.dataset import fetch_gaussian_params_from_CHOIR
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
        skip_connections: bool = False,
    ):
        super().__init__()
        self.skip_connections = skip_connections
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
        bs, ctx_len = x.shape[0], (x.shape[1] if len(x.shape) == 4 else 1)
        t = t.view(bs * ctx_len, -1)
        x = x.view(x.shape[0], -1)
        time_embed = self.time_mlp(self.time_encoder(t))
        x1 = self.input_layer(x)
        x2 = self.block_1(x1, t_emb=time_embed, y_emb=y, debug=debug)
        x3 = self.block_2(x2, t_emb=time_embed, y_emb=y, debug=debug)
        x4 = self.block_3(x3, t_emb=time_embed, y_emb=y, debug=debug)
        x5 = self.block_4(x4, t_emb=time_embed, y_emb=y, debug=debug)
        if self.skip_connections:
            x5 = x2 + x5
        # x5 = x4
        x6 = self.output_layer(x5)
        output_shape = (*input_shape[:-2], self.output_dim // 3, 3)
        return x6.view(output_shape)


class ContactMLPResNetBackboneModel(torch.nn.Module):
    def __init__(
        self,
        time_encoder: torch.nn.Module,
        input_dim: int,
        output_dim_hand: int,
        temporal_dim: int,
        contact_dim: int,
        contact_hidden_dim: int = 1024,
        hidden_dim: int = 1024,
        context_dim: Optional[int] = None,
        skip_connections: bool = False,
    ):
        super().__init__()
        self.skip_connections = skip_connections
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
        self.contact_input_layer = torch.nn.Sequential(
            torch.nn.Linear(
                contact_dim + 128,
                contact_hidden_dim,
            ),
            torch.nn.GroupNorm(32, contact_hidden_dim),
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
        self.feat_prop = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_dim, 128),
        )
        self.contact_1 = temporal_res_block(
            dim_in=contact_hidden_dim, dim_out=contact_hidden_dim
        )
        self.contact_2 = temporal_res_block(
            dim_in=contact_hidden_dim, dim_out=contact_hidden_dim
        )
        self.contact_3 = temporal_res_block(
            dim_in=contact_hidden_dim, dim_out=contact_hidden_dim
        )
        self.contact_4 = temporal_res_block(
            dim_in=contact_hidden_dim, dim_out=contact_hidden_dim
        )
        self.output_layer = torch.nn.Linear(hidden_dim, output_dim_hand)
        self.contacts_output_layer = torch.nn.Linear(contact_hidden_dim, contact_dim)
        self.output_dim_hand = output_dim_hand

    def set_anchor_indices(self, anchor_indices: torch.Tensor):
        pass

    def forward(
        self,
        x: torch.Tensor,
        contacts: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        debug=False,
    ) -> torch.Tensor:
        input_shape_x, input_shape_contacts = x.shape, contacts.shape
        bs, ctx_len = x.shape[0], (x.shape[1] if len(x.shape) == 4 else 1)
        t = t.view(bs * ctx_len, -1)
        x = x.view(bs * ctx_len, -1)
        contacts = contacts.view(bs * ctx_len, -1)
        time_embed = self.time_mlp(self.time_encoder(t))
        x1 = self.input_layer(x)
        x2 = self.block_1(x1, t_emb=time_embed, y_emb=y, debug=debug)
        x3 = self.block_2(x2, t_emb=time_embed, y_emb=y, debug=debug)
        x4 = self.block_3(x3, t_emb=time_embed, y_emb=y, debug=debug)
        x5 = self.block_4(x4, t_emb=time_embed, y_emb=y, debug=debug)
        if self.skip_connections:
            x5 = x2 + x5
        x6 = self.output_layer(x5)

        c1 = self.contact_input_layer(torch.cat((contacts, self.feat_prop(x3)), dim=-1))
        c2 = self.contact_1(c1, t_emb=time_embed, y_emb=y, debug=debug)
        c3 = self.contact_2(c2, t_emb=time_embed, y_emb=y, debug=debug)
        c4 = self.contact_3(c3, t_emb=time_embed, y_emb=y, debug=debug)
        c5 = self.contact_4(c4, t_emb=time_embed, y_emb=y, debug=debug)
        if self.skip_connections:
            c5 = c2 + c5
        c6 = self.contacts_output_layer(c5)
        kp_shape = (*input_shape_x[:-2], self.output_dim_hand // 3, 3)
        return x6.view(kp_shape), c6.view(input_shape_contacts)


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
        # self.fc2 = torch.nn.Linear(512, 256)
        # self.bn2 = torch.nn.BatchNorm1d(256)
        # self.drop2 = torch.nn.Dropout(0.4)
        self.fc3 = torch.nn.Linear(512, embed_dim)

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
        # x = self.drop2(torch.nn.functional.gelu(self.bn2(self.fc2(x))))
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
        x = x.view(b, t, -1)
        x = self.ff(x)
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
        self.output_layer = torch.nn.Linear(hidden_dim, embed_dim)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = x.view(x.shape[0], -1)
        x1 = self.input_layer(x)
        x2 = self.block_1(x1)
        x3 = self.block_2(x2)
        x4 = self.block_3(x3)
        return self.output_layer(x4)


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
        context_channels: Optional[int | Tuple[int]] = None,
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
        n_scales = 5  # Corresponds to this U-Net architecture, not a hyperparameter.
        same_context_channels = (
            type(context_channels) == int if context_channels is not None else False
        )
        if not same_context_channels:
            assert (
                len(context_channels) == n_scales
            ), "Context channels must be specified for each of the 5 scales, or be an integer."
        context_channels = (
            ([context_channels] * n_scales)
            if same_context_channels
            else context_channels
        )
        self.multi_scale_encoder = not same_context_channels
        # ========= Partials =========
        down_conv_block = partial(
            TemporalConvDownBlock,
            temporal_channels=temporal_dim,
            normalization=normalization,
            norm_groups=norm_groups,
            pooling=pooling,
        )
        up_conv_block = partial(
            TemporalConvUpBlock,
            temporal_channels=temporal_dim,
            normalization=normalization,
            norm_groups=norm_groups,
            interpolate=True,
        )
        identity_conv_block = partial(
            TemporalConvIdentityBlock,
            temporal_channels=temporal_dim,
            normalization=normalization,
            norm_groups=norm_groups,
        )
        spatial_transformer = partial(
            SpatialTransformer,
            n_heads=8,
            dim_heads=32,
            dropout=0.1,
            gated_ff=False,
            norm_groups=norm_groups,
        )
        # ========= Layers =========
        dim_heads = 32
        self.identity1 = identity_conv_block(
            channels_in=self.choir_dim,
            channels_out=64,
            norm_groups=min(16, norm_groups),
            context_channels=context_channels[0],
        )
        self.down1 = down_conv_block(
            channels_in=64,
            channels_out=64,
            norm_groups=min(16, norm_groups),
            context_channels=context_channels[1],
        )
        # self.self_attn_1 = (
        # spatial_transformer(
        # in_channels=64, n_heads=64 // dim_heads, dim_heads=dim_heads
        # )
        # if use_self_attention
        # else None
        # )
        self.down2 = down_conv_block(
            channels_in=64, channels_out=128, context_channels=context_channels[2]
        )
        self.self_attn_2 = (
            spatial_transformer(
                in_channels=128, n_heads=128 // dim_heads, dim_heads=dim_heads
            )
            if use_self_attention
            else None
        )
        self.down3 = down_conv_block(
            channels_in=128, channels_out=256, context_channels=context_channels[3]
        )
        self.self_attn_3 = (
            spatial_transformer(
                in_channels=256, n_heads=256 // dim_heads, dim_heads=dim_heads
            )
            if use_self_attention
            else None
        )
        self.tunnel1 = identity_conv_block(
            channels_in=256, channels_out=256, context_channels=context_channels[4]
        )
        self.self_attn_4 = (
            spatial_transformer(
                in_channels=256, n_heads=256 // dim_heads, dim_heads=dim_heads
            )
            if use_self_attention
            else None
        )
        self.tunnel2 = identity_conv_block(
            channels_in=256, channels_out=256, context_channels=context_channels[4]
        )
        self.up1 = up_conv_block(
            channels_in=512,
            channels_out=128,
            output_padding=output_paddings[0],
            context_channels=context_channels[3],
        )
        self.self_attn_5 = (
            spatial_transformer(
                in_channels=128, n_heads=128 // dim_heads, dim_heads=dim_heads
            )
            if use_self_attention
            else None
        )
        self.up2 = up_conv_block(
            channels_in=256,
            channels_out=64,
            output_padding=output_paddings[1],
            context_channels=context_channels[2],
        )
        self.self_attn_6 = (
            spatial_transformer(
                in_channels=64, n_heads=64 // dim_heads, dim_heads=dim_heads
            )
            if use_self_attention
            else None
        )
        self.up3 = up_conv_block(
            channels_in=128,
            channels_out=64,
            output_padding=output_paddings[2],
            context_channels=context_channels[1],
        )
        # self.self_attn_7 = (
        # spatial_transformer(
        # in_channels=64, n_heads=64 // dim_heads, dim_heads=dim_heads
        # )
        # if use_self_attention
        # else None
        # )
        self.identity3 = identity_conv_block(
            channels_in=64,
            channels_out=64,
            norm_groups=min(16, norm_groups),
            context_channels=context_channels[0],
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
            if self.multi_scale_encoder:
                raise NotImplementedError(
                    "Generation not implemented for multi-scale encoder"
                )
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
        if self.multi_scale_encoder:
            assert x.shape[0] == y[0].shape[0], "Batch size mismatch between x and y"
        else:
            assert x.shape[0] == y.shape[0], "Batch size mismatch between x and y"
        if not self.multi_scale_encoder:
            y = [y] * 5
        t_embed = self.time_mlp(self.time_encoder(t))
        x1 = self.identity1(x, t_embed, context=y[0], debug=debug)
        x2 = self.down1(x1, t_embed, context=y[1], debug=debug)
        # x2 = self.self_attn_1(x2) if self.use_self_attn else x2
        x3 = self.down2(x2, t_embed, context=y[2], debug=debug)
        x3 = self.self_attn_2(x3) if self.use_self_attn else x3
        x4 = self.down3(x3, t_embed, context=y[3], debug=debug)
        x4 = self.self_attn_3(x4) if self.use_self_attn else x4
        x5 = self.tunnel1(x4, t_embed, context=y[4], debug=debug)
        x5 = self.self_attn_4(x5) if self.use_self_attn else x5
        x6 = self.tunnel2(x5, t_embed, context=y[4], debug=debug)
        # The output of the final downsampling layer is concatenated with the output of the final
        # tunnel layer because they have the same shape H and W. Then we upscale those features and
        # conctenate the upscaled features with the output of the previous downsampling layer, and
        # so on.
        x7 = self.up1(torch.cat((x6, x4), dim=1), t_embed, context=y[3], debug=debug)
        x7 = self.self_attn_5(x7) if self.use_self_attn else x7
        x8 = self.up2(torch.cat((x7, x3), dim=1), t_embed, context=y[2], debug=debug)
        x8 = self.self_attn_6(x8) if self.use_self_attn else x8
        x9 = self.up3(torch.cat((x8, x2), dim=1), t_embed, context=y[1], debug=debug)
        # x9 = self.self_attn_7(x9) if self.use_self_attn else x9
        x10 = self.identity3(x9, t_embed, context=y[0], debug=debug)
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


class ContactUNetBackboneModel(torch.nn.Module):
    def __init__(
        self,
        time_encoder: torch.nn.Module,
        bps_grid_len: int,
        input_dim: int,
        output_dim: int,
        contacts_dim: int,
        temporal_dim: int,
        contacts_hidden_dim: int = 1024,
        contacts_skip_connections: bool = False,
        pooling: str = "avg",
        normalization: str = "batch",
        norm_groups: int = 16,
        output_paddings: Tuple[int] = (1, 1, 1, 1),
        context_channels: Optional[int | Tuple[int]] = None,
        use_self_attention: bool = False,
        interpolate: bool = False,
    ):
        super().__init__()
        self.grid_len = bps_grid_len
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.time_encoder = time_encoder
        self.use_self_attn = use_self_attention
        self.anchor_indices = None
        self.contacts_skip_connections = contacts_skip_connections
        # 32 anchors assigned randomly to each BPS point, hence the repetition (see section on anchor assignment in the paper).
        self.n_gaussian_params, self.n_anchors = contacts_dim, 32
        self.n_repeats = (self.grid_len**3) // self.n_anchors
        self.time_mlp = torch.nn.Sequential(
            torch.nn.Linear(temporal_dim, temporal_dim),
            torch.nn.GELU(),
            torch.nn.Linear(temporal_dim, temporal_dim),
        )
        n_scales = 5  # Corresponds to this U-Net architecture, not a hyperparameter.
        same_context_channels = (
            type(context_channels) == int if context_channels is not None else False
        )
        if not same_context_channels:
            assert (
                len(context_channels) == n_scales
            ), "Context channels must be specified for each of the 5 scales, or be an integer."
        context_channels = (
            ([context_channels] * n_scales)
            if same_context_channels
            else context_channels
        )
        self.multi_scale_encoder = not same_context_channels
        # ========= Partials =========
        temporal_res_block = partial(
            TemporalResBlock,
            dim_in=contacts_hidden_dim,
            dim_out=contacts_hidden_dim,
            n_norm_groups=32,
            temporal_dim=temporal_dim,
            y_dim=context_channels[-1],
        )
        down_conv_block = partial(
            TemporalConvDownBlock,
            temporal_channels=temporal_dim,
            normalization=normalization,
            norm_groups=norm_groups,
            pooling=pooling,
        )
        up_conv_block = partial(
            TemporalConvUpBlock,
            temporal_channels=temporal_dim,
            normalization=normalization,
            norm_groups=norm_groups,
            interpolate=interpolate,
        )
        identity_conv_block = partial(
            TemporalConvIdentityBlock,
            temporal_channels=temporal_dim,
            normalization=normalization,
            norm_groups=norm_groups,
        )
        spatial_transformer = partial(
            SpatialTransformer,
            n_heads=8,
            dim_heads=32,
            dropout=0.1,
            gated_ff=False,
            norm_groups=norm_groups,
        )
        # ======== Contact layers =========
        self.contacts_dim = contacts_dim
        self.feat_prop = torch.nn.Dropout(0.1)
        self.contact_1 = temporal_res_block(
            dim_in=contacts_dim * self.n_anchors + 256
        )  # input x + embedding from unet
        self.contact_2 = temporal_res_block()
        self.contact_3 = temporal_res_block()
        self.contact_4 = temporal_res_block()
        self.contact_output = torch.nn.Linear(
            contacts_hidden_dim, contacts_dim * self.n_anchors
        )
        # ========= Feature fusion =========
        # self.feature_fusion_contacts_to_dist = MultiHeadSpatialAttention(
        # q_dim=256,
        # k_dim=256,
        # v_dim=256,
        # n_heads=8,
        # dim_head=32,
        # p_dropout=0.1,
        # )
        # self.feature_fusion_dist_to_contacts = MultiHeadSpatialAttention(
        # q_dim=256,
        # k_dim=256,
        # v_dim=256,
        # n_heads=8,
        # dim_head=32,
        # p_dropout=0.1,
        # )
        # ========= Layers =========
        dim_heads = 32
        self.identity1 = identity_conv_block(
            channels_in=self.input_dim,
            channels_out=64,
            norm_groups=min(16, norm_groups),
            context_channels=context_channels[0],
        )
        self.down1 = down_conv_block(
            channels_in=64,
            channels_out=64,
            norm_groups=min(16, norm_groups),
            context_channels=context_channels[1],
        )
        # self.self_attn_1 = (
        # spatial_transformer(
        # in_channels=64, n_heads=64 // dim_heads, dim_heads=dim_heads
        # )
        # if use_self_attention
        # else None
        # )
        self.down2 = down_conv_block(
            channels_in=64, channels_out=128, context_channels=context_channels[2]
        )
        self.self_attn_2 = (
            spatial_transformer(
                in_channels=128, n_heads=128 // dim_heads, dim_heads=dim_heads
            )
            if use_self_attention
            else None
        )
        self.down3 = down_conv_block(
            channels_in=128, channels_out=256, context_channels=context_channels[3]
        )
        self.self_attn_3 = (
            spatial_transformer(
                in_channels=256, n_heads=256 // dim_heads, dim_heads=dim_heads
            )
            if use_self_attention
            else None
        )
        self.tunnel1 = identity_conv_block(
            channels_in=256, channels_out=256, context_channels=context_channels[4]
        )
        self.self_attn_4 = (
            spatial_transformer(
                in_channels=256, n_heads=256 // dim_heads, dim_heads=dim_heads
            )
            if use_self_attention
            else None
        )
        self.tunnel2 = identity_conv_block(
            channels_in=256, channels_out=256, context_channels=context_channels[4]
        )
        self.up1 = up_conv_block(
            channels_in=512,
            channels_out=128,
            output_padding=output_paddings[0],
            context_channels=context_channels[3],
        )
        self.self_attn_5 = (
            spatial_transformer(
                in_channels=128, n_heads=128 // dim_heads, dim_heads=dim_heads
            )
            if use_self_attention
            else None
        )
        self.up2 = up_conv_block(
            channels_in=256,
            channels_out=64,
            output_padding=output_paddings[1],
            context_channels=context_channels[2],
        )
        self.self_attn_6 = (
            spatial_transformer(
                in_channels=64, n_heads=64 // dim_heads, dim_heads=dim_heads
            )
            if use_self_attention
            else None
        )
        self.up3 = up_conv_block(
            channels_in=128,
            channels_out=64,
            output_padding=output_paddings[2],
            context_channels=context_channels[1],
        )
        # self.self_attn_7 = (
        # spatial_transformer(
        # in_channels=64, n_heads=64 // dim_heads, dim_heads=dim_heads
        # )
        # if use_self_attention
        # else None
        # )
        self.identity3 = identity_conv_block(
            channels_in=64,
            channels_out=64,
            norm_groups=min(16, norm_groups),
            context_channels=context_channels[0],
        )
        self.out_conv = torch.nn.Conv3d(64, self.output_dim, 1, padding=0, stride=1)

    def set_anchor_indices(self, anchor_indices: torch.Tensor):
        self.anchor_indices = anchor_indices

    def forward(
        self,
        x_udf: torch.Tensor,
        x_contacts: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.tensor] = None,
        debug: bool = False,
    ) -> torch.Tensor:
        udf_input_shape, contacts_input_shape = x_udf.shape, x_contacts.shape
        bs, ctx_len = x_udf.shape[0], (x_udf.shape[1] if len(x_udf.shape) == 4 else 1)
        if ctx_len > 1:
            if self.multi_scale_encoder:
                raise NotImplementedError(
                    "Generation not implemented for multi-scale encoder"
                )
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
            self.input_dim,
        )
        t = t.view(bs * ctx_len, -1)
        x_udf = x_udf.view(*comp_shape).permute(0, 4, 1, 2, 3)
        x_contacts = x_contacts.view(bs * ctx_len, self.contacts_dim * self.n_anchors)
        assert x_udf.shape[0] == t.shape[0], "Batch size mismatch between x and t"
        if self.multi_scale_encoder:
            assert (
                x_udf.shape[0] == y[0].shape[0]
            ), "Batch size mismatch between x and y"
        else:
            assert x_udf.shape[0] == y.shape[0], "Batch size mismatch between x and y"
        if not self.multi_scale_encoder:
            y = [y] * 5
        t_embed = self.time_mlp(self.time_encoder(t))
        x1 = self.identity1(x_udf, t_embed, context=y[0], debug=debug)
        x2 = self.down1(x1, t_embed, context=y[1], debug=debug)
        # x2 = self.self_attn_1(x2) if self.use_self_attn else x2
        x3 = self.down2(x2, t_embed, context=y[2], debug=debug)
        x3 = self.self_attn_2(x3) if self.use_self_attn else x3
        x4 = self.down3(x3, t_embed, context=y[3], debug=debug)
        x4 = self.self_attn_3(x4) if self.use_self_attn else x4
        x5 = self.tunnel1(x4, t_embed, context=y[4], debug=debug)
        x5 = self.self_attn_4(x5) if self.use_self_attn else x5
        x6 = self.tunnel2(x5, t_embed, context=y[4], debug=debug)
        # ========= Feature fusion =========
        unet_features = x6.mean(dim=(2, 3, 4))
        unet_features = self.feat_prop(unet_features)
        c1 = self.contact_1(
            torch.cat((x_contacts, unet_features), dim=-1),
            t_emb=t_embed,
            y_emb=y[4].view(bs * ctx_len, -1),
            debug=debug,
        )
        c2 = self.contact_2(
            c1, t_emb=t_embed, y_emb=y[4].view(bs * ctx_len, -1), debug=debug
        )
        # cx6 = x6 + self.feature_fusion_contacts_to_dist(
        # q=x6, k=c3[..., None, None, None], v=c3[..., None, None, None]
        # )
        # c3x = c3 + self.feature_fusion_dist_to_contacts(
        # q=c3[..., None, None, None], k=x6, v=x6
        # ).view(bs * ctx_len, c3.shape[1])
        # The output of the final downsampling layer is concatenated with the output of the final
        # tunnel layer because they have the same shape H and W. Then we upscale those features and
        # conctenate the upscaled features with the output of the previous downsampling layer, and
        # so on.
        x7 = self.up1(torch.cat((x6, x4), dim=1), t_embed, context=y[3], debug=debug)
        x7 = self.self_attn_5(x7) if self.use_self_attn else x7
        c3 = self.contact_3(
            c2, t_emb=t_embed, y_emb=y[4].view(bs * ctx_len, -1), debug=debug
        )
        if self.contacts_skip_connections:
            c3 = c3 + c1
        x8 = self.up2(torch.cat((x7, x3), dim=1), t_embed, context=y[2], debug=debug)
        x8 = self.self_attn_6(x8) if self.use_self_attn else x8
        c4 = self.contact_4(
            c3, t_emb=t_embed, y_emb=y[4].view(bs * ctx_len, -1), debug=debug
        )
        if self.contacts_skip_connections:
            c4 = c4 + c2
        x9 = self.up3(torch.cat((x8, x2), dim=1), t_embed, context=y[1], debug=debug)
        # x9 = self.self_attn_7(x9) if self.use_self_attn else x9
        x10 = self.identity3(x9, t_embed, context=y[0], debug=debug)
        c5 = self.contact_output(c4)
        comp_output_shape = (
            bs * ctx_len,
            self.output_dim,
            self.grid_len,
            self.grid_len,
            self.grid_len,
        )
        output_shape = (
            *udf_input_shape[:-1],
            self.output_dim,
        )
        output = (
            self.out_conv(x10)
            .permute(0, 2, 3, 4, 1)
            .reshape(comp_output_shape)
            .view(output_shape)
        )
        return output, c5.view(contacts_input_shape)


class ContactUNetBackboneModel_legacy(UNetBackboneModel):
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
        context_channels: Optional[int | Tuple[int]] = None,
        use_self_attention: bool = False,
        no_decoding: bool = False,
    ):
        super().__init__(
            time_encoder,
            bps_grid_len,
            choir_dim,
            output_channels,
            temporal_dim,
            pooling,
            normalization,
            norm_groups,
            output_paddings,
            context_channels=context_channels,
            use_self_attention=use_self_attention,
        )
        self._no_decoding = no_decoding
        self.anchor_indices = None
        # 32 anchors assigned randomly to each BPS point, hence the repetition (see section on anchor assignment in the paper).
        self.n_gaussian_params, self.n_anchors = 9, 32
        self.n_repeats = (self.grid_len**3) // self.n_anchors
        # self.contacts_decoder = torch.nn.Sequential(
        # torch.nn.BatchNorm1d(self.n_anchors),
        # torch.nn.Linear(self.n_gaussian_params * self.n_repeats, 2048),
        # torch.nn.GELU(),
        # torch.nn.Linear(2048, self.n_gaussian_params),
        # )

    def set_anchor_indices(self, anchor_indices: torch.Tensor):
        self.anchor_indices = anchor_indices

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.tensor] = None,
        debug: bool = False,
    ) -> torch.Tensor:
        assert (
            self.anchor_indices is not None
        ), "Anchor indices must be set before forward pass."
        x = super().forward(x, t, y, debug)
        if self._no_decoding:
            return x
        # Now, we'll decode the gaussian parameters by groups of N_REPEAT * 9,
        # where N_REPEAT = (bps_grid_len ** 3) / 32. We can batch everything together in a batch of
        # 32 * N_REPEAT, and then reshape the output to the original shape. This way, the MLP
        # aggregator only has to aggregate input features of the SAME MANO ANCHOR. Much more
        # efficient than pooling bps_grid_len ** 3 * 9 features into 32 * 9 parameters.
        # anchor_indices <- (BPS_GRID_LEN ** 3,) indicate which anchor each BPS point is assigned to.
        # We want gaussian_param_feats to have shape (B, N_REPEAT, 32, 9), where B is the batch size.
        # So, we need to permute gaussian_param_feats to go from [0, 7, 9, 31, 2, 4, ...] in the
        # second dimension to [0, 0, 0, ..., 1, 1, 1, ..., 31, 31, 31] in the second dimension. For
        # this, we'll use anchor_indices:
        # We can just sort the indices and then use the sorted indices to permute the second dimension.
        # This way, we'll have all the features of the same anchor together.
        # We'll then reshape the tensor to (B, N_REPEAT, 32, 9) and pass it through the MLP.
        gaussian_params = fetch_gaussian_params_from_CHOIR(
            x,
            self.anchor_indices,
            self.n_repeats,
            self.n_anchors,
            choir_includes_obj=False,
        )

        gaussian_params = gaussian_params.mean(dim=-2)  # Simple mean pooling as a PoC
        # shape = gaussian_params.shape
        # gaussian_params = gaussian_params.view(*shape[:-2] , self.n_gaussian_params * self.n_repeats)
        # gaussian_params = self.contacts_decoder(gaussian_params)  # (B, 32, 9)

        # Finally, we'll concatenate the gaussian parameters to the original output tensor.
        # By doing this, we also undo the sorting of the indices to get the original order of the
        # BPS points.
        x = torch.cat(
            (
                x[..., 0].unsqueeze(
                    -1
                ),  # The first channel is the CHOIR field (hand dist only though)
                gaussian_params[:, self.anchor_indices],
            ),
            dim=-1,
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
        # self.self_attn_1 = (
        # spatial_transformer(
        # in_channels=64, n_heads=64 // dim_heads, dim_heads=dim_heads
        # )
        # if use_self_attention
        # else None
        # )
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
        # if self.use_self_attn:
        # x = self.self_attn_1(x)
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


class MultiScaleResnetEncoderModel(torch.nn.Module):
    def __init__(
        self,
        bps_grid_len: int,
        choir_dim: int,
        embed_channels: int,
        pooling: str = "max",
        normalization: str = "batch",
        norm_groups: int = 16,
        use_self_attention: bool = False,
        feature_pooling: str = "none",
    ):
        super().__init__()
        self.grid_len = bps_grid_len
        self.choir_dim = choir_dim
        self.use_self_attn = use_self_attention
        identity_conv_block = partial(
            ConvIdentityBlock,
            normalization=normalization,
            norm_groups=norm_groups,
        )
        down_conv_block = partial(
            ConvDownBlock,
            normalization=normalization,
            norm_groups=norm_groups,
            pooling=pooling,
        )
        spatial_transformer = partial(
            SpatialTransformer,
            n_heads=8,
            dim_heads=32,
            dropout=0.0,
            gated_ff=False,
            norm_groups=norm_groups,
        )
        dim_heads = 32
        self.identity = identity_conv_block(
            choir_dim,
            16,
        )
        self.down1 = down_conv_block(
            16,
            32,
        )
        # self.self_attn_1 = (
        # spatial_transformer(
        # in_channels=64, n_heads=64 // dim_heads, dim_heads=dim_heads
        # )
        # if use_self_attention
        # else None
        # )
        self.down2 = down_conv_block(
            32,
            64,
        )
        self.down3 = down_conv_block(
            64,
            128,
        )
        # self.self_attn_3 = (
        #    spatial_transformer(
        #        in_channels=256, n_heads=256 // dim_heads, dim_heads=dim_heads
        #    )
        #    if use_self_attention
        #    else None
        # )
        self.out_identity = identity_conv_block(
            128,
            embed_channels,
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
        self.out_conv = torch.nn.Conv3d(
            embed_channels, embed_channels, 1, padding=0, stride=1
        )
        self.channel_pooling = feature_pooling

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
        x = x.view(
            bs * n * ctx_len,
            self.grid_len,
            self.grid_len,
            self.grid_len,
            self.choir_dim,
        ).permute(0, 4, 1, 2, 3)
        l0 = self.identity(x, debug=debug)
        l1 = self.down1(l0, debug=debug)
        l2 = self.down2(l1, debug=debug)
        l3 = self.down3(l2, debug=debug)
        l4 = self.out_identity(l3, debug=debug)
        l4 = self.out_conv(l4)
        for level in [l0, l1, l2, l3, l4]:
            level = level.view(bs * n, ctx_len, *level.shape[1:]).squeeze(
                dim=1
            )  # For now we'll try with 1 context frame TODO

        feature_levels = [l0, l1, l2, l3, l4]
        if self.channel_pooling != "none":
            for i in range(len(feature_levels)):
                if self.channel_pooling == "max":
                    feature_levels[i] = (
                        feature_levels[i].max(dim=1, keepdim=True).values
                    )
                elif self.channel_pooling == "avg":
                    feature_levels[i] = feature_levels[i].mean(dim=1, keepdim=True)
        return tuple(feature_levels)


class TransformerEncoderModel(torch.nn.Module):
    def __init__(
        self,
        bps_grid_len: int,
        choir_dim: int,
        embed_channels: int,
        num_layers: int = 4,
        patch_size: int = 4,
        n_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        global_pooling: bool = False,
        non_linear_projection: bool = False,
        spatialize_patches: bool = False,
    ):
        super().__init__()
        self.grid_len = bps_grid_len
        self.choir_dim = choir_dim
        self.patch_size = patch_size
        self.transformer_encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=embed_channels,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,  # (batch, seq, feature) where seq is basis points and feature is choir_dim
            ),
            num_layers=num_layers,
        )
        n_patches = (self.grid_len // self.patch_size) ** 3
        self.pos_encoder = SinusoidalPosEncoder(
            max_positions=n_patches, model_dim=embed_channels
        )
        self.pooling = (
            torch.nn.Sequential(
                torch.nn.Linear(embed_channels * n_patches, 2048),
                torch.nn.GELU(),
                torch.nn.Linear(2048, embed_channels),
            )
            if global_pooling
            else None
        )
        self.linear_projection = torch.nn.Sequential(
            torch.nn.Linear(choir_dim * self.patch_size**3, embed_channels),
            torch.nn.GELU() if non_linear_projection else torch.nn.Identity(),
        )
        self.embed_dim = embed_channels
        # if spatialize_patches:
        # cb_rt = int(round(embed_channels ** (1 / 3)))
        # assert cb_rt**3 == embed_channels, (
        # "Embedding dimension must be a perfect cube for spatialized patches. "
        # + f"Got {embed_channels} with cube root {cb_rt}"
        # )
        self.spatialize_patches = spatialize_patches

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
        if debug:
            print(f"Input shape: {x.shape}")
        n_patches = (self.grid_len // self.patch_size) ** 3
        if debug:
            print(f"Patch indices: {torch.arange(n_patches)})")
        if debug:
            print(
                f"Pos indices shape: {torch.arange(n_patches).unsqueeze(0).repeat(bs * n * ctx_len, 1).to(x.device).shape}"
            )
        pos_embed = self.pos_encoder(
            torch.arange(n_patches)
            .unsqueeze(0)
            .repeat(bs * n * ctx_len, 1)
            .to(x.device),
            flatten_output=False,
        )
        if debug:
            print(f"Positional embedding shape: {pos_embed.shape}")
        x = x.view(
            bs * n * ctx_len,
            self.grid_len,
            self.grid_len,
            self.grid_len,
            self.choir_dim,
        ).permute(0, 4, 1, 2, 3)
        if debug:
            print(f"Reshaped shape: {x.shape}")
        # Make patches of size self.patch_size x self.patch_size x self.patch_size:
        # (Output should be (bs * n * ctx_len, choir_dim, self.patch_size, self.patch_size,
        # self.patch_size, n_patches)).
        x = (
            x.unfold(
                2, self.patch_size, self.patch_size
            )  # Along the 2nd dimension (x-axis): square patch
            .unfold(
                3, self.patch_size, self.patch_size
            )  # Along the 3rd dimension (y-axis): square patch
            .unfold(
                4, self.patch_size, self.patch_size
            )  # Along the 4th dimension (z-axis): square patch
        )
        if debug:
            print(f"Unfolded shape: {x.shape}")
        x = x.reshape(
            bs * n * ctx_len,
            self.choir_dim,
            self.patch_size,
            self.patch_size,
            self.patch_size,
            n_patches,
        )
        if debug:
            print(f"Patches shape: {x.shape}")
        # Flatten the patches:
        x = x.permute(0, 5, 1, 2, 3, 4).flatten(
            start_dim=2
        )  # (bs * n * ctx_len, n_patches, choir_dim * self.patch_size ** 3)
        if debug:
            print(f"Flattened shape: {x.shape}")
        x = self.linear_projection(x)
        if debug:
            print(f"Projected shape: {x.shape}")
        x = x + pos_embed
        x = self.transformer_encoder(x)
        if debug:
            print(f"Transformer encoded shape: {x.shape}")
        if self.pooling is not None:
            x = self.pooling(
                x.flatten(start_dim=1)
            )  # Concatenate all patches and pool them
            if debug:
                print(f"Pooled shape: {x.shape}")
            x = x.view(bs * n, ctx_len, -1, 1, 1, 1).squeeze(
                dim=1
            )  # For now we'll try with 1 context frame TODO
        elif self.spatialize_patches:
            # patch_dim = int(math.cbrt(self.embed_dim))
            # x is (bs * n * ctx_len, n_patches, d_model)
            x = x.permute(0, 2, 1)  # (bs * n * ctx_len, d_model, n_patches)
            # We'll use n_patches as the spatial dimension. It's basically the same as the
            # non-spatialized version haha.
            x = x.view(
                bs * n * ctx_len, self.embed_dim, n_patches, 1, 1
            )  # D, H, W is basically n_patches because it's linear
            if debug:
                print(f"Spatialized shape: {x.shape}")
            x = x.view(bs * n, ctx_len, *x.shape[1:])
            x = x.squeeze(dim=1)
        else:
            x = (
                x.permute(0, 2, 1)  # (bs * n * ctx_len, d_model, n_patches)
                .view(
                    bs * n,
                    ctx_len,
                    self.embed_dim,
                    self.patch_size,  # Each patch is d=256 and we have 64 patches.
                    self.patch_size,
                    self.patch_size,
                )
                .squeeze(dim=1)
            )
        if debug:
            print(f"Output shape: {x.shape}")
        return x
