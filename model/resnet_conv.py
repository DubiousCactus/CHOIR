#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
2D ResNet building blocks using convolutions.
"""


from functools import partial
from typing import Optional, Tuple

import torch

from model.attention import MultiHeadSpatialAttention


class InterpolateUpsampleLayer(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int = 2,
        mode: str = "nearest",
        conv_kernel: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.conv = (
            torch.nn.Conv3d(
                in_channels, out_channels, kernel_size=conv_kernel, padding=1
            )
            if conv_kernel is not None
            else torch.nn.Identity()
        )
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.interpolate(
            x, scale_factor=self.scale_factor, mode=self.mode
        )
        x = self.conv(x)
        return x


class TemporalResidualBlock(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        channels_in: int,
        channels_out: int,
        temporal_dim: Optional[int] = None,
        norm_groups: int = 16,
        normalization: str = "group",
        rescaling: str = "none",
        strides: Tuple[int, int] = (1, 1, 1, 1),
        paddings: Tuple[int, int] = (1, 0, 1, 0),
        kernels: Tuple[int, int] = (3, 0, 3, 1),
        output_padding: int = 0,
        context_channels: Optional[int] = None,
        x_attn_heads: int = 16,
        x_attn_head_dim: int = 32,
        pooling: str = "none",
        interpolate: bool = False,
        p_dropout: float = 0.1,
    ):
        super().__init__()
        assert dim in (2, 3), "Only 2D and 3D convolutions are supported"
        assert rescaling in ("none", "down", "up"), "Invalid rescaling"
        assert pooling in ("none", "avg", "max"), "Invalid pooling"
        assert not (
            pooling != "none" and interpolate
        ), "Cannot have both pooling and interpolating"
        kwargs = (
            {"output_padding": output_padding}
            if not interpolate and rescaling == "up"
            else {}
        )
        assert normalization in (
            "group",
            "batch",
        ), "Only group and batch normalization are supported"
        self.is_2d = dim == 2
        if rescaling == "up" and not interpolate:
            conv = torch.nn.ConvTranspose2d if dim == 2 else torch.nn.ConvTranspose3d
        else:
            conv = torch.nn.Conv2d if dim == 2 else torch.nn.Conv3d
        pool = None
        if pooling != "none":
            pool = (
                (torch.nn.AvgPool2d if dim == 2 else torch.nn.AvgPool3d)
                if pooling == "avg"
                else (torch.nn.MaxPool2d if dim == 2 else torch.nn.MaxPool3d)
            )
        norm = (
            partial(torch.nn.GroupNorm, min(norm_groups, channels_out))
            if normalization == "group"
            else (torch.nn.BatchNorm2d if dim == 2 else torch.nn.BatchNorm3d)
        )

        # ================== INPUT CONVOLUTION =====================
        self.norm1 = norm(channels_out)
        self.conv1 = conv(
            channels_in,
            channels_out,
            kernels[0],
            padding=paddings[0],
            stride=strides[0],
        )
        self.nonlin = torch.nn.SiLU()
        # ==========================================================
        # ================== UP/DOWN/NO SAMPLING ===================
        if rescaling == "down":
            self.up_or_down_sample = (
                conv(
                    channels_out,
                    channels_out,
                    kernels[1],
                    padding=paddings[1],
                    stride=strides[1],
                )
                if pooling == "none"
                else pool(kernel_size=2, stride=2)
            )
            self.up_or_down_sample_input = (
                conv(
                    channels_in,
                    channels_in,
                    kernels[1],
                    padding=paddings[1],
                    stride=strides[1],
                )
                if pooling == "none"
                else pool(kernel_size=2, stride=2)
            )
        elif rescaling == "up":
            self.up_or_down_sample = (
                InterpolateUpsampleLayer(channels_out, channels_out, conv_kernel=None)
                if interpolate
                else conv(
                    channels_out,
                    channels_out,
                    kernels[1],
                    padding=paddings[1],
                    stride=strides[1],
                    **kwargs,
                )
            )
            self.up_or_down_sample_input = (
                InterpolateUpsampleLayer(channels_in, channels_in, conv_kernel=None)
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
            self.up_or_down_sample_input = torch.nn.Identity()

        # ================= OUTPUT CONVOLUTION =====================
        self.norm2 = norm(channels_out)
        self.conv2 = conv(channels_out, channels_out, kernels[0], padding=paddings[0])
        self.out_activation = torch.nn.SiLU()
        self.out_norm = norm(channels_out)
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
            MultiHeadSpatialAttention(
                q_dim=channels_out,
                k_dim=context_channels,
                v_dim=context_channels,
                n_heads=x_attn_heads,
                dim_head=x_attn_head_dim,
                p_dropout=p_dropout,
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
        _x = x
        if t_emb is not None:
            scale, shift = (
                self.temporal_projection(t_emb)[..., None, None].chunk(2, dim=1)
                if self.is_2d
                else self.temporal_projection(t_emb)[..., None, None, None].chunk(
                    2, dim=1
                )
            )
        x = self.conv1(x)
        x = self.norm1(x)
        if t_emb is not None:
            x = x * (scale + 1) + shift
        x = self.nonlin(x)
        x = self.up_or_down_sample(x)
        x = self.norm2(x)
        x = self.conv2(x)
        if t_emb is not None:
            x = x * (scale + 1) + shift
        if context is not None:
            assert t_emb is not None
            x = self.out_activation(
                self.out_norm(
                    x + self.skip_connection(self.up_or_down_sample_input(_x))
                )
            )
            # TODO: Move the following OUTSIDE this module, into its own module.
            x = x + self.cross_attention(q=x, k=context, v=context)
        else:
            x = self.out_activation(
                self.out_norm(
                    x + self.skip_connection(self.up_or_down_sample_input(_x))
                )
            )
        return x


class TemporalIdentityResidualBlock(TemporalResidualBlock):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        temporal_channels: int,
        norm_groups: int = 16,
        normalization: str = "group",
        dim: int = 3,
        context_channels: Optional[int] = None,
    ):
        super().__init__(
            dim,
            channels_in,
            channels_out,
            temporal_channels,
            norm_groups,
            normalization,
            context_channels=context_channels,
        )


class TemporalDownScaleResidualBlock(TemporalResidualBlock):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        temporal_channels: int,
        pooling: str = "avg",
        norm_groups: int = 16,
        normalization: str = "group",
        dim: int = 3,
        context_channels: Optional[int] = None,
    ):
        super().__init__(
            dim,
            channels_in,
            channels_out,
            temporal_channels,
            norm_groups,
            normalization,
            rescaling="down",
            context_channels=context_channels,
            pooling=pooling,
            strides=(1, 2, 1, 2) if pooling == "none" else (1, 1, 1, 1),
            paddings=(1, 1, 1, 0),
            kernels=(3, 3, 3, 1),
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
        dim: int = 3,
        context_channels: Optional[int] = None,
    ):
        super().__init__(
            dim,
            channels_in,
            channels_out,
            temporal_channels,
            norm_groups,
            normalization,
            rescaling="up",
            interpolate=interpolate,
            context_channels=context_channels,
            strides=(1, 2, 1, 1) if not interpolate else (1, 1, 1, 1),
            paddings=(1, 1, 1, 0),
            kernels=(3, 3, 3, 1),
            output_padding=output_padding,
        )


class IdentityResidualBlock(TemporalResidualBlock):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        norm_groups: int = 16,
        normalization: str = "group",
        dim: int = 3,
    ):
        super().__init__(
            dim,
            channels_in,
            channels_out,
            temporal_dim=None,
            norm_groups=norm_groups,
            normalization=normalization,
        )


class DownScaleResidualBlock(TemporalResidualBlock):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        pooling: str = "avg",
        norm_groups: int = 16,
        normalization: str = "group",
        dim: int = 3,
    ):
        super().__init__(
            dim,
            channels_in,
            channels_out,
            temporal_dim=None,
            norm_groups=norm_groups,
            normalization=normalization,
            rescaling="down",
            pooling=pooling,
            strides=(1, 2, 1, 2) if pooling == "none" else (1, 1, 1, 1),
            paddings=(1, 1, 1, 0),
            kernels=(3, 3, 3, 1),
        )


class UpScaleResidualBlock(TemporalResidualBlock):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        interpolate: bool = True,
        output_padding: int = 0,
        norm_groups: int = 16,
        normalization: str = "group",
        dim: int = 3,
        context_channels: Optional[int] = None,
    ):
        super().__init__(
            dim,
            channels_in,
            channels_out,
            temporal_dim=None,
            norm_groups=norm_groups,
            normalization=normalization,
            interpolate=interpolate,
            rescaling="up",
            strides=(1, 2, 1, 1) if not interpolate else (1, 1, 1, 1),
            paddings=(1, 1, 1, 0),
            kernels=(3, 3, 3, 1),
            output_padding=output_padding,
        )
