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
import os
import os.path as osp
import pickle
from typing import Any, Dict, List, Tuple

import torch
from bps_torch.tools import sample_sphere_uniform
from hydra.utils import get_original_cwd
from metabatch import TaskSet


class BaseDataset(TaskSet, abc.ABC):
    def __init__(
        self,
        dataset_name: str,
        bps_dim: int,
        max_views_per_grasp: int,
        noisy_samples_per_grasp: int,
        rescale: str,
        center_on_object_com: bool,
        remap_bps_distances: bool,
        exponential_map_w: float,
        augment: bool,
        split: str,
        tiny: bool = False,
        seed: int = 0,
        debug: bool = False,
    ) -> None:
        # This feels super hacky because I hadn't designed metabatch to be used this way. It was
        # only meant to be used with the traditional context + target paradigm. This will work but
        # requires min_pts=1 and max_ctx_pts=1, as well as using n_target. Don't change any of
        # these defaults or it will break for this case.
        # TODO: Fix perturbation_level=0
        assert noisy_samples_per_grasp >= max_views_per_grasp
        super().__init__(
            min_pts=1,
            max_ctx_pts=max_views_per_grasp,
            max_tgt_pts=noisy_samples_per_grasp,
            total_tgt_pts=noisy_samples_per_grasp,
            eval=False,
            predict_full_target=False,
            predict_full_target_during_eval=False,
        )
        self._mm_unit = 1.0
        self._augment = augment and split == "train"
        self._bps_dim = bps_dim
        self._cache_dir = osp.join(
            get_original_cwd(), "data", f"{dataset_name}_preprocessed"
        )
        if not osp.isdir(self._cache_dir):
            os.makedirs(self._cache_dir)
        bps_path = osp.join(self._cache_dir, f"bps_{self._bps_dim}.pkl")
        if osp.isfile(bps_path):
            with open(bps_path, "rb") as f:
                bps = pickle.load(f)
        else:
            bps = sample_sphere_uniform(
                n_points=self._bps_dim, n_dims=3, radius=1.0, random_seed=1995
            ).cpu()
            with open(bps_path, "wb") as f:
                pickle.dump(bps, f)
        self._bps = bps
        self._center_on_object_com = center_on_object_com
        self._rescale = rescale
        self._remap_bps_distances = remap_bps_distances
        self._exponential_map_w = exponential_map_w
        self._debug = debug
        (
            objects_w_contacts,
            grasps,
            dataset_name,
        ) = self._load_objects_and_grasps(tiny, split, seed=seed)
        self._sample_paths: List[List[str]] = self._load(
            split, objects_w_contacts, grasps, dataset_name
        )

    @property
    def base_unit(self) -> float:
        return self._mm_unit

    @property
    def remap_bps_distances(self) -> bool:
        return self._remap_bps_distances

    @property
    def exponential_map_w(self) -> float:
        return self._exponential_map_w

    @property
    def bps_dim(self) -> int:
        return self._bps_dim

    @property
    def bps(self) -> torch.Tensor:
        return self._bps

    @abc.abstractmethod
    def _load(
        self,
        split: str,
        objects: Dict,
        objects_w_contacts: List,
        grasps: List,
        dataset_name: str,
    ) -> List[List[str]]:
        """
        Returns a list of noisy grasp sequences. Each grasp sequence is a list of noisy grasp paths.
        """
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

    def __gettask__(
        self, idx: int, n_context: int, n_target: int
    ) -> Tuple[List[Tuple], List[Tuple]]:
        noisy_grasp_samples = self._sample_paths[idx]
        assert noisy_grasp_samples is not None
        assert n_context <= len(noisy_grasp_samples)
        # Randomly sample n_context grasps from the grasp sequence (ignore n_target since we're not
        # doing meta-learning here):
        samples_paths = torch.randperm(len(noisy_grasp_samples))[:n_context]
        samples_paths = [
            noisy_grasp_samples[i] for i in samples_paths
        ]  # Paths to grasp sample/label pairs
        samples, labels = [], []
        for sample_path in samples_paths:
            with open(sample_path, "rb") as f:
                sample, label = pickle.load(f)
            samples.append(sample)
            labels.append(label)
        return torch.stack(samples), torch.stack(labels)

    # def __getitem__(self, idx: int) -> Tuple[Any, Any]:
    # with open(self._sample_paths[idx], "rb") as f:
    # sample, label = pickle.load(f)
    # return sample, label
