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
import pickle
from typing import Any, List, Tuple

from torch.utils.data import Dataset


class BaseDataset(Dataset, abc.ABC):
    def __init__(
        self,
        dataset_root: str,
        augment: bool,
        split: str,
        tiny: bool = False,
        scaling: str = "hand_object",
        seed: int = 0,
        debug: bool = False,
    ) -> None:
        super().__init__()
        self._mm_unit = 1.0
        self._augment = augment and split == "train"
        self._scaling = scaling
        self._debug = debug
        objects, grasps, dataset_name = self._load_objects_and_grasps(
            tiny, split, seed=seed
        )
        self._sample_paths = self._load(
            dataset_root, tiny, split, objects, grasps, dataset_name
        )

    @abc.abstractmethod
    def _load(
        self,
        dataset_root: str,
        tiny: bool,
        split: str,
        objects: List,
        grasps: List,
        dataset_name: str,
    ) -> List[str]:
        # Implement this
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self._sample_paths)

    def disable_augs(self) -> None:
        self._augment = False

    def _load_objects_and_grasps(
        self, tiny: bool, split: str, seed: int = 0
    ) -> Tuple[List[Any], List[Any], str]:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        with open(self._sample_paths[idx], "rb") as f:
            sample, label = pickle.load(f)
        return sample, label
