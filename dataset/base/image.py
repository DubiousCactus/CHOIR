#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Base dataset for images.
"""


import abc
from typing import Optional, Tuple, Union

import albumentations as A
import torch
from torchvision.io.image import read_image
from torchvision.transforms import transforms

from dataset.base import BaseDataset


class ImageDataset(BaseDataset, abc.ABC):
    IMAGE_NET_MEAN, IMAGE_NET_STD = ([], [])
    COCO_MEAN, COCO_STD = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    IMG_SIZE = (32, 32)

    def __init__(
        self,
        dataset_root: str,
        split: str,
        img_size: Optional[tuple] = None,
        augment: bool = False,
        normalize: bool = False,
        tiny: bool = False,
    ) -> None:
        super().__init__(dataset_root, augment, normalize, split, tiny=tiny)
        self._img_size = self.IMG_SIZE if img_size is None else img_size
        self._transforms = transforms.Compose(
            [
                transforms.Resize(self._img_size),
            ]
        )
        self._normalization = transforms.Normalize(
            self.IMAGE_NET_MEAN, self.IMAGE_NET_STD
        )
        self._augs = A.Compose(
            [
                A.RandomCropFromBorders(),
                A.RandomBrightnessContrast(),
                A.RandomGamma(),
            ]
        )

    @abc.abstractmethod
    def _load(
        self, dataset_root: str, tiny: bool, split: Optional[str] = None
    ) -> Tuple[Union[dict, list, torch.Tensor], Union[dict, list, torch.Tensor]]:
        # Implement this
        raise NotImplementedError

    def __getitem__(self, index: int):
        """
        This should be common to all image datasets!
        Override if you need something else.
        """
        # ==== Load image and apply transforms ===
        img = read_image(self._samples[index])  # Returns a Tensor
        img = self._transforms(img)
        if self._normalize:
            img = self._normalization(img)
        if self._augment:
            img = self._augs(image=img)
        # ==== Load label and apply transforms ===
        label = self._labels[index]
        return img, label
