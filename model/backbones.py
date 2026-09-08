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




