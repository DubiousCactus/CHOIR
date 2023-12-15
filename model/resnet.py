#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
ResNet building blocks.
"""


from typing import Tuple

import torch


class ResidualBlock(torch.nn.Module):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        norm_groups: int = 8,
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
        self.nonlin = torch.nn.GELU()
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
        self.out_activation = torch.nn.GELU()
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
        norm_groups: int = 8,
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
        self.nonlin = torch.nn.GELU()
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
        self.out_activation = torch.nn.GELU()
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

    def forward(
        self, x: torch.Tensor, t_emb: torch.Tensor, debug: bool = False
    ) -> torch.Tensor:
        def print_debug(str):
            nonlocal debug
            if debug:
                print(str)

        _x = x
        scale, shift = (
            self.temporal_projection(t_emb).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
        )
        print_debug(f"Starting with x.shape = {x.shape}")
        x = self.conv1(x)
        print_debug(f"After conv1, x.shape = {x.shape}")
        x = self.norm1(x)
        x *= (scale + 1) + shift
        x = self.nonlin(x)
        print_debug(f"Temb is {t_emb.shape}")
        print_debug(f"Temb projected is {self.temporal_projection(t_emb).shape}")
        x = self.conv2(x)
        print_debug(f"After conv2, x.shape = {x.shape}")
        x = self.norm2(x)
        x *= (scale + 1) + shift
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
        norm_groups: int = 8,
        normalization: str = "group",
    ):
        super().__init__(
            channels_in, channels_out, temporal_channels, norm_groups, normalization
        )


class TemporalDownScaleResidualBlock(TemporalResidualBlock):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        temporal_channels: int,
        pooling: bool = False,  # TODO
        norm_groups: int = 8,
        normalization: str = "group",
    ):
        super().__init__(
            channels_in,
            channels_out,
            temporal_channels,
            norm_groups,
            normalization,
            strides=(2, 1, 2),
            paddings=(1, 1, 0),
            kernels=(3, 3, 1),
            conv=torch.nn.Conv2d,
        )


class TemporalUpScaleResidualBlock(TemporalResidualBlock):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        temporal_channels: int,
        upsampling: bool = False,  # TODO
        output_padding: int = 0,
        norm_groups: int = 8,
        normalization: str = "group",
    ):
        super().__init__(
            channels_in,
            channels_out,
            temporal_channels,
            norm_groups,
            normalization,
            strides=(1, 2, 2),
            paddings=(1, 1, 0),
            kernels=(3, 3, 1),
            output_padding=output_padding,
            conv=torch.nn.ConvTranspose2d,
        )


class IdentityResidualBlock(ResidualBlock):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        norm_groups: int = 8,
        normalization: str = "group",
    ):
        super().__init__(channels_in, channels_out, norm_groups, normalization)


class DownScaleResidualBlock(ResidualBlock):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        pooling: bool = False,  # TODO
        norm_groups: int = 8,
        normalization: str = "group",
    ):
        super().__init__(
            channels_in,
            channels_out,
            norm_groups,
            normalization,
            strides=(2, 1, 2),
            paddings=(1, 1, 0),
            kernels=(3, 3, 1),
            conv=torch.nn.Conv2d,
        )


class UpScaleResidualBlock(ResidualBlock):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        upsampling: bool = False,  # TODO
        output_padding: int = 0,
        norm_groups: int = 8,
        normalization: str = "group",
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
            conv=torch.nn.ConvTranspose2d,
        )
