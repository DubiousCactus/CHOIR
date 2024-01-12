#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
2D ResNet building blocks using convolutions.
"""


from typing import Optional, Tuple

import torch

from model.attention import MultiHeadAttention


class TemporalResidualBlock(torch.nn.Module):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        temporal_dim: Optional[int] = None,
        norm_groups: int = 16,
        normalization: str = "group",
        input_norm: bool = True,
        strides: Tuple[int, int] = (1, 1, 1, 1),
        paddings: Tuple[int, int] = (1, 0, 1, 0),
        kernels: Tuple[int, int] = (3, 0, 3, 1),
        output_padding: int = 0,
        context_channels: Optional[int] = None,
        x_attn_heads: int = 8,
        x_attn_head_dim: int = 64,
        pooling: bool = False,
        interpolate: bool = False,
        p_dropout: float = 0.0,
        conv=torch.nn.Conv2d,
        pool=torch.nn.AvgPool2d,
        drop=torch.nn.Dropout2d,
    ):
        super().__init__()
        assert not (
            pooling and interpolate
        ), "Cannot have both pooling and interpolating"
        kwargs = (
            {"output_padding": output_padding}
            if conv in (torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d)
            else {}
        )
        assert normalization in (
            "group",
            "batch",
        ), "Only group and batch normalization are supported"
        self.is_2d = conv in (torch.nn.Conv2d, torch.nn.ConvTranspose2d)

        # ================== INPUT CONVOLUTION =====================
        norm1 = (
            torch.nn.GroupNorm(min(norm_groups, channels_in), channels_in)
            if normalization == "group"
            else (
                torch.nn.BatchNorm2d(channels_in)
                if self.is_2d
                else torch.nn.BatchNorm3d(channels_in)
            )
        )
        self.in_layers = torch.nn.Sequential(
            norm1 if input_norm else torch.nn.Identity(),
            torch.nn.SiLU(),
            conv(
                channels_in,
                channels_out,
                kernels[0],
                padding=paddings[0],
                stride=strides[0],
            ),
        )
        # ==========================================================
        # ================== UP/DOWN/NO SAMPLING ===================
        upsampling = (
            conv in (torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d) or interpolate
        )
        downsampling = not interpolate and (
            pooling or strides[1] > 1 or paddings[1] != 0
        )
        if downsampling:
            self.up_or_down_sample = (
                conv(
                    channels_in,
                    channels_in,
                    kernels[1],
                    padding=paddings[1],
                    stride=strides[1],
                    **kwargs,
                )
                if not pooling
                else pool(kernel_size=2, stride=2)
            )
        elif upsampling:
            self.up_or_down_sample = (
                lambda x: torch.nn.functional.interpolate(
                    x, scale_factor=2, mode="nearest"
                )
                if interpolate
                else conv(
                    channels_in,
                    channels_in,
                    kernels[1],
                    padding=paddings[1],
                    stride=strides[1],
                    **kwargs,
                )
            )
        else:
            self.up_or_down_sample = torch.nn.Identity()

        norm2 = (
            torch.nn.GroupNorm(norm_groups, channels_out)
            if normalization == "group"
            else (
                torch.nn.BatchNorm2d(channels_out)
                if self.is_2d
                else torch.nn.BatchNorm3d(channels_out)
            )
        )
        # ==========================================================
        # ================== OUTPUT CONVOLUTION ====================
        self.out_layers = torch.nn.Sequential(
            norm2,
            torch.nn.SiLU(),
            drop(p=p_dropout),
            conv(
                channels_out,
                channels_out,
                kernels[2],
                padding=paddings[2],
                stride=strides[2],
            ),
        )
        # ==========================================================
        # ================== TEMPORAL PROJECTION ===================
        self.temporal_projection = (
            torch.nn.Sequential(
                torch.nn.SiLU(),
                torch.nn.Linear(
                    temporal_dim,
                    channels_out * 2,
                ),
            )
            if temporal_dim is not None
            else torch.nn.Identity()
        )
        # ==========================================================
        # ================== SKIP CONNECTION =======================
        self.skip_connection = (
            conv(
                channels_in,
                channels_out,
                kernels[3],
                padding=paddings[3],
                stride=strides[3],
                bias=False,
            )
            if (channels_in != channels_out or strides[3] != 1 or paddings[3] != 0)
            else torch.nn.Identity()
        )
        # ==========================================================
        # ================== CROSS-ATTENTION =======================
        self.cross_attention = (
            MultiHeadAttention(
                q_dim=channels_out,
                k_dim=context_channels,
                v_dim=context_channels,
                n_heads=x_attn_heads,
                dim_head=x_attn_head_dim,
                p_dropout=0.0,
            )
            if context_channels is not None
            else None
        )
        # ==========================================================

    def forward(
        self,
        x: torch.Tensor,
        t_emb: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        debug: bool = False,
    ) -> torch.Tensor:
        def print_debug(str):
            nonlocal debug
            if debug:
                print(str)

        _x = x
        if t_emb is not None:
            scale, shift = (
                self.temporal_projection(t_emb)[..., None, None].chunk(2, dim=1)
                if self.is_2d
                else self.temporal_projection(t_emb)[..., None, None, None].chunk(
                    2, dim=1
                )
            )
        # print_debug("=========================================")
        # print_debug(f"Starting with x.shape = {x.shape}")
        # print_debug(f"scale and shift shapes: {scale.shape}, {shift.shape}")
        rest, conv = self.in_layers[:-1], self.in_layers[-1]
        x = rest(x)
        x = self.up_or_down_sample(x)
        # print_debug(f"After up_or_down_sample, x.shape = {x.shape}")
        x = conv(x)
        # print_debug(f"After input layers, x.shape = {x.shape}")
        norm, rest = self.out_layers[0], self.out_layers[1:]
        x = norm(x)
        if t_emb is not None:
            x = (
                x * (scale + 1) + shift
            )  # Normalize before scale/shift as in OpenAI's code
        x = rest(x)
        # print_debug(f"After output layers, x.shape = {x.shape}")
        if context is not None:
            # TODO: Implement as a separate block like OpenAI
            x = x + self.cross_attention(q=x, k=context, v=context)
            # x = x * (scale + 1) + shift  # Normalize before scale/shift as in OpenAI's code
        # print_debug(
        # f"Adding _x of shape {_x.shape} (rescaled to {self.skip_connection(self.up_or_down_sample(_x)).shape}) to x of shape {x.shape}"
        # )
        # print_debug("=========================================")
        return x + self.skip_connection(self.up_or_down_sample(_x))


class TemporalIdentityResidualBlock(TemporalResidualBlock):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        temporal_channels: int,
        norm_groups: int = 16,
        normalization: str = "group",
        conv: str = "2d",
        context_channels: Optional[int] = None,
        input_norm: bool = True,
    ):
        assert conv in ("2d", "3d"), "Only 2D and 3D convolutions are supported"
        super().__init__(
            channels_in,
            channels_out,
            temporal_channels,
            norm_groups,
            normalization,
            context_channels=context_channels,
            input_norm=input_norm,
            conv=torch.nn.Conv2d if conv == "2d" else torch.nn.Conv3d,
            drop=torch.nn.Dropout2d if conv == "2d" else torch.nn.Dropout3d,
        )


class TemporalDownScaleResidualBlock(TemporalResidualBlock):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        temporal_channels: int,
        pooling: bool = True,
        norm_groups: int = 16,
        normalization: str = "group",
        conv: str = "2d",
        context_channels: Optional[int] = None,
    ):
        assert conv in ("2d", "3d"), "Only 2D and 3D convolutions are supported"
        super().__init__(
            channels_in,
            channels_out,
            temporal_channels,
            norm_groups,
            normalization,
            context_channels=context_channels,
            pooling=pooling,
            strides=(1, 2, 1, 2) if not pooling else (1, 1, 1, 1),
            paddings=(1, 1, 1, 0),
            kernels=(3, 3, 3, 1),
            conv=torch.nn.Conv2d if conv == "2d" else torch.nn.Conv3d,
            pool=torch.nn.AvgPool2d if conv == "2d" else torch.nn.AvgPool3d,
            drop=torch.nn.Dropout2d if conv == "2d" else torch.nn.Dropout3d,
        )


class TemporalUpScaleResidualBlock(TemporalResidualBlock):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        temporal_channels: int,
        interpolate: bool = True,
        output_padding: int = 0,
        norm_groups: int = 16,
        normalization: str = "group",
        conv: str = "2d",
        context_channels: Optional[int] = None,
    ):
        assert conv in ("2d", "3d"), "Only 2D and 3D convolutions are supported"
        super().__init__(
            channels_in,
            channels_out,
            temporal_channels,
            norm_groups,
            normalization,
            interpolate=interpolate,
            context_channels=context_channels,
            strides=(1, 2, 1, 1) if not interpolate else (1, 1, 1, 1),
            paddings=(1, 1, 1, 0),
            kernels=(3, 3, 3, 1),
            output_padding=output_padding,
            conv=(
                torch.nn.ConvTranspose2d if conv == "2d" else torch.nn.ConvTranspose3d
            )
            if not interpolate
            else (torch.nn.Conv2d if conv == "2d" else torch.nn.Conv3d),
            drop=torch.nn.Dropout2d if conv == "2d" else torch.nn.Dropout3d,
        )


class IdentityResidualBlock(TemporalResidualBlock):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        norm_groups: int = 16,
        normalization: str = "group",
        conv: str = "2d",
        input_norm: bool = True,
    ):
        super().__init__(
            channels_in,
            channels_out,
            temporal_dim=None,
            norm_groups=norm_groups,
            normalization=normalization,
            input_norm=input_norm,
            conv=torch.nn.Conv2d if conv == "2d" else torch.nn.Conv3d,
            drop=torch.nn.Dropout2d if conv == "2d" else torch.nn.Dropout3d,
        )


class DownScaleResidualBlock(TemporalResidualBlock):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        pooling: bool = False,
        norm_groups: int = 16,
        normalization: str = "group",
        conv: str = "2d",
    ):
        super().__init__(
            channels_in,
            channels_out,
            temporal_dim=None,
            norm_groups=norm_groups,
            normalization=normalization,
            pooling=pooling,
            strides=(1, 2, 1, 2) if not pooling else (1, 1, 1, 1),
            paddings=(1, 1, 1, 0),
            kernels=(3, 3, 3, 1),
            conv=torch.nn.Conv2d if conv == "2d" else torch.nn.Conv3d,
            pool=torch.nn.AvgPool2d if conv == "2d" else torch.nn.AvgPool3d,
            drop=torch.nn.Dropout2d if conv == "2d" else torch.nn.Dropout3d,
        )


class UpScaleResidualBlock(TemporalResidualBlock):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        upsampling: bool = False,  # TODO
        output_padding: int = 0,
        norm_groups: int = 16,
        normalization: str = "group",
        conv: str = "2d",
        context_channels: Optional[int] = None,
    ):
        super().__init__(
            channels_in,
            channels_out,
            temporal_dim=None,
            norm_groups=norm_groups,
            normalization=normalization,
            strides=(1, 2, 1, 2),
            paddings=(1, 1, 1, 0),
            kernels=(3, 3, 3, 1),
            output_padding=output_padding,
            conv=torch.nn.ConvTranspose2d if conv == "2d" else torch.nn.ConvTranspose3d,
        )
