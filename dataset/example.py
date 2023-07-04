#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Example dataset inheriting from the base ImageDataset class.
This is mostly used to test the framework.
"""

from typing import Optional, Tuple

import torch

from dataset.base.image import ImageDataset


class ExampleDataset(ImageDataset):
    IMG_SIZE = (32, 32)

    def __init__(
        self,
        dataset_root: str,
        split: str,
        img_dim: Optional[int] = None,
        augment: bool = False,
        normalize: bool = False,
        tiny: bool = False,
    ) -> None:
        self._img_dim = self.IMG_SIZE[0] if img_dim is None else img_dim
        super().__init__(
            dataset_root, split, (img_dim, img_dim), augment, normalize, tiny
        )

    def _load(
        self, dataset_root: str, tiny: bool, split: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.rand(10000, self._img_dim, self._img_dim), torch.rand(10000, 8)

    def __getitem__(self, index: int):
        return self._samples[index], self._labels[index]
