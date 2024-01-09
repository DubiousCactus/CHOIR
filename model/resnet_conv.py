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


class ResidualBlock(torch.nn.Module):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        norm_groups: int = 16,
        normalization: str = "group",
        strides: Tuple[int, int] = (1, 1, 1),
        paddings: Tuple[int, int] = (1, 1, 0),
        kernels: Tuple[int, int] = (3, 3, 1),
        output_padding: int = 0,
        conv=torch.nn.Conv2d,
    ):
        super().__init__()
        kwargs = (
            {"output_padding": output_padding}
            if conv == torch.nn.ConvTranspose2d
            else {}
        )
        self.conv1 = conv(
            channels_in,
            channels_out,
            kernels[0],
            padding=paddings[0],
            stride=strides[0],
        )
        self.norm1 = (
            torch.nn.GroupNorm(norm_groups, channels_out)
            if normalization == "group"
            else torch.nn.BatchNorm2d(channels_out)
        )
        self.nonlin = torch.nn.SiLU()
        self.conv2 = conv(
            channels_out,
            channels_out,
            kernels[1],
            padding=paddings[1],
            stride=strides[1],
            **kwargs,
        )
        self.norm2 = (
            torch.nn.GroupNorm(norm_groups, channels_out)
            if normalization == "group"
            else torch.nn.BatchNorm2d(channels_out)
        )
        self.out_activation = torch.nn.SiLU()
        self.residual_scaling = (
            conv(
                channels_in,
                channels_out,
                kernels[2],
                padding=paddings[2],
                stride=strides[2],
                bias=False,
                **kwargs,
            )
            if (channels_in != channels_out or strides[2] != 1 or paddings[2] != 0)
            else torch.nn.Identity()
        )

    def forward(self, x: torch.Tensor, debug: bool = False) -> torch.Tensor:
        def print_debug(str):
            nonlocal debug
            if debug:
                print(str)

        _x = x
        print_debug(f"Starting with x.shape = {x.shape}")
        x = self.conv1(x)
        print_debug(f"After conv1, x.shape = {x.shape}")
        x = self.norm1(x)
        x = self.nonlin(x)
        x = self.conv2(x)
        print_debug(f"After conv2, x.shape = {x.shape}")
        x = self.norm2(x)
        print_debug(
            f"Adding _x of shape {_x.shape} (rescaled to {self.residual_scaling(_x).shape}) to x of shape {x.shape}"
        )
        return self.out_activation(x + self.residual_scaling(_x))


class TemporalResidualBlock(torch.nn.Module):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        temporal_dim: int,
        norm_groups: int = 16,
        normalization: str = "group",
        strides: Tuple[int, int] = (1, 1, 1),
        paddings: Tuple[int, int] = (1, 1, 0),
        kernels: Tuple[int, int] = (3, 3, 1),
        output_padding: int = 0,
        context_channels: Optional[int] = None,
        x_attn_heads: int = 8,
        x_attn_head_dim: int = 64,
        conv=torch.nn.Conv2d,
    ):
        super().__init__()
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
        self.conv1 = conv(
            channels_in,
            channels_out,
            kernels[0],
            padding=paddings[0],
            stride=strides[0],
        )
        self.norm1 = (
            torch.nn.GroupNorm(norm_groups, channels_out)
            if normalization == "group"
            else (
                torch.nn.BatchNorm2d(channels_out)
                if self.is_2d
                else torch.nn.BatchNorm3d(channels_out)
            )
        )
        self.nonlin = torch.nn.SiLU()
        self.conv2 = conv(
            channels_out,
            channels_out,
            kernels[1],
            padding=paddings[1],
            stride=strides[1],
            **kwargs,
        )
        self.norm2 = (
            torch.nn.GroupNorm(norm_groups, channels_out)
            if normalization == "group"
            else (
                torch.nn.BatchNorm2d(channels_out)
                if self.is_2d
                else torch.nn.BatchNorm3d(channels_out)
            )
        )
        self.out_activation = torch.nn.SiLU()
        self.temporal_projection = torch.nn.Linear(
            temporal_dim,
            channels_out * 2,
        )
        self.residual_scaling = (
            conv(
                channels_in,
                channels_out,
                kernels[2],
                padding=paddings[2],
                stride=strides[2],
                bias=False,
                **kwargs,
            )
            if (channels_in != channels_out or strides[2] != 1 or paddings[2] != 0)
            else torch.nn.Identity()
        )
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

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        debug: bool = False,
    ) -> torch.Tensor:
        def print_debug(str):
            nonlocal debug
            if debug:
                print(str)

        _x = x
        scale, shift = (
            self.temporal_projection(t_emb)[..., None, None].chunk(2, dim=1)
            if self.is_2d
            else self.temporal_projection(t_emb)[..., None, None, None].chunk(2, dim=1)
        )
        print_debug(f"Starting with x.shape = {x.shape}")
        print_debug(f"scale and shift shapes: {scale.shape}, {shift.shape}")
        x = self.conv1(x)
        print_debug(f"After conv1, x.shape = {x.shape}")
        # if context is not None:
        # x = x + self.cross_attention(q=x, k=context, v=context)
        x = self.norm1(x)
        x = x * (scale + 1) + shift  # Normalize before scale/shift as in OpenAI's code
        x = self.nonlin(x)
        print_debug(f"Temb is {t_emb.shape}")
        print_debug(f"Temb projected is {self.temporal_projection(t_emb).shape}")
        x = self.conv2(x)
        print_debug(f"After conv2, x.shape = {x.shape}")
        if context is not None:
            x = x + self.cross_attention(q=x, k=context, v=context)
        x = self.norm2(x)
        x = x * (scale + 1) + shift  # Normalize before scale/shift as in OpenAI's code
        print_debug(
            f"Adding _x of shape {_x.shape} (rescaled to {self.residual_scaling(_x).shape}) to x of shape {x.shape}"
        )
        return self.out_activation(x + self.residual_scaling(_x))


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
    ):
        assert conv in ("2d", "3d"), "Only 2D and 3D convolutions are supported"
        super().__init__(
            channels_in,
            channels_out,
            temporal_channels,
            norm_groups,
            normalization,
            context_channels=context_channels,
            conv=torch.nn.Conv2d if conv == "2d" else torch.nn.Conv3d,
        )


class TemporalDownScaleResidualBlock(TemporalResidualBlock):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        temporal_channels: int,
        pooling: bool = False,  # TODO
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
            strides=(2, 1, 2),
            paddings=(1, 1, 0),
            kernels=(3, 3, 1),
            conv=torch.nn.Conv2d if conv == "2d" else torch.nn.Conv3d,
        )


class TemporalUpScaleResidualBlock(TemporalResidualBlock):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        temporal_channels: int,
        upsampling: bool = False,  # TODO
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
            context_channels=context_channels,
            strides=(1, 2, 2),
            paddings=(1, 1, 0),
            kernels=(3, 3, 1),
            output_padding=output_padding,
            conv=torch.nn.ConvTranspose2d if conv == "2d" else torch.nn.ConvTranspose3d,
        )


class IdentityResidualBlock(ResidualBlock):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        norm_groups: int = 16,
        normalization: str = "group",
        conv: str = "2d",
        context_channels: Optional[int] = None,
    ):
        super().__init__(
            channels_in,
            channels_out,
            norm_groups,
            normalization,
            conv=torch.nn.Conv2d if conv == "2d" else torch.nn.Conv3d,
        )


class DownScaleResidualBlock(ResidualBlock):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        pooling: bool = False,  # TODO
        norm_groups: int = 16,
        normalization: str = "group",
        conv: str = "2d",
        context_channels: Optional[int] = None,
    ):
        super().__init__(
            channels_in,
            channels_out,
            norm_groups,
            normalization,
            strides=(2, 1, 2),
            paddings=(1, 1, 0),
            kernels=(3, 3, 1),
            conv=torch.nn.Conv2d if conv == "2d" else torch.nn.Conv3d,
        )


class UpScaleResidualBlock(ResidualBlock):
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
            norm_groups,
            normalization,
            strides=(1, 2, 2),
            paddings=(1, 1, 0),
            kernels=(3, 3, 1),
            output_padding=output_padding,
            conv=torch.nn.ConvTranspose2d if conv == "2d" else torch.nn.ConvTranspose3d,
        )
