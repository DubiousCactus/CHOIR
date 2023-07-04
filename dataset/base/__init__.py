#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Base dataset.
In this file you may implement other base datasets that share the same characteristics and which
need the same data loading + transformation pipeline. The specificities of loading the data or
transforming it may be extended through class inheritance in a specific dataset file.
"""


import abc
from typing import Tuple, Union

from torch.utils.data import Dataset


class BaseDataset(Dataset, abc.ABC):
    def __init__(
        self,
        dataset_root: str,
        augment: bool,
        normalize: bool,
        split: str,
        tiny: bool = False,
    ) -> None:
        super().__init__()
        self._samples, self._labels = self._load(dataset_root, tiny, split)
        self._augment = augment and split == "train"
        self._normalize = normalize

    @abc.abstractmethod
    def _load(
        self, dataset_root: str, tiny: bool, split: str
    ) -> Tuple[Union[dict, list], Union[dict, list]]:
        # Implement this
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self._samples)

    def disable_augs(self) -> None:
        self._augment = False
