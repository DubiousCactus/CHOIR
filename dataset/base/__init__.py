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
import itertools
import os
import os.path as osp
import pickle
from typing import Any, List, Tuple

import blosc
import torch
from bps_torch.tools import sample_sphere_uniform
from hydra.utils import get_original_cwd
from metabatch import TaskSet


class BaseDataset(TaskSet, abc.ABC):
    def __init__(
        self,
        split: str,
        validation_objects: int,
        test_objects: int,
        perturbation_level: int,
        obj_ptcld_size: int,
        bps_dim: int,
        noisy_samples_per_grasp: int,
        max_views_per_grasp: int,
        right_hand_only: bool,
        center_on_object_com: bool,
        tiny: bool,
        augment: bool,
        n_augs: int,
        seed: int,
        debug: bool,
        rescale: str,
        remap_bps_distances: bool,
        exponential_map_w: float,
        dataset_name: str,
    ) -> None:
        # This feels super hacky because I hadn't designed metabatch to be used this way. It was
        # only meant to be used with the traditional context + target paradigm. This will work but
        # requires min_pts=1 and max_ctx_pts=1, as well as using n_target. Don't change any of
        # these defaults or it will break for this case.
        noisy_samples_per_grasp = (
            1 if perturbation_level == 0 else noisy_samples_per_grasp
        )
        assert noisy_samples_per_grasp >= max_views_per_grasp
        super().__init__(
            min_pts=1,
            max_ctx_pts=max_views_per_grasp,
            max_tgt_pts=noisy_samples_per_grasp
            + 1,  # This is actually irrelevant in our case! Just needs to be higher than max_ctx_pts
            total_tgt_pts=noisy_samples_per_grasp,
            eval=False,
            predict_full_target=False,
            predict_full_target_during_eval=False,
        )
        self._mm_unit = 1.0
        self._split = split
        self._validation_objects = validation_objects
        self._test_objects = test_objects
        self._obj_ptcld_size = obj_ptcld_size
        self._right_hand_only = right_hand_only
        self._perturbation_level = perturbation_level
        self._augment = augment and split == "train"
        self._n_augs = n_augs
        self._bps_dim = bps_dim
        self._noisy_samples_per_grasp = noisy_samples_per_grasp
        # self._perturbations = [] # TODO: Implement
        self._cache_dir = osp.join(
            get_original_cwd(), "data", f"{dataset_name}_preprocessed"
        )
        if not osp.isdir(self._cache_dir):
            os.makedirs(self._cache_dir)
        bps_path = osp.join(
            get_original_cwd(), "data", f"bps_{self._bps_dim}_{rescale}-rescaled.pkl"
        )
        if osp.isfile(bps_path):
            with open(bps_path, "rb") as f:
                bps = pickle.load(f)
        else:
            bps = sample_sphere_uniform(
                n_points=self._bps_dim,
                n_dims=3,
                radius=0.2 if rescale == "none" else 0.6,
                random_seed=1995,
            ).cpu()
            with open(bps_path, "wb") as f:
                pickle.dump(bps, f)
        self._bps = bps
        self._center_on_object_com = center_on_object_com
        self._rescale = rescale
        self._remap_bps_distances = remap_bps_distances
        self._exponential_map_w = exponential_map_w
        self._observations_number = None
        self._n_combinations = None
        self._combinations = None
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
    def is_right_hand_only(self) -> bool:
        return self._right_hand_only

    @property
    def bps(self) -> torch.Tensor:
        return self._bps

    def set_observations_number(self, n: int) -> None:
        # This SO thread explains the problem and the (super simple) solution:
        # https://stackoverflow.com/questions/27974126/get-all-n-choose-k-combinations-of-length-n
        self._observations_number = n
        self._combinations = list(
            itertools.combinations(range(self._noisy_samples_per_grasp), n)
        )
        self._n_combinations = len(self._combinations)

    @abc.abstractmethod
    def _load(
        self,
        split: str,
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
        if self._split == "test":
            assert (
                self._observations_number is not None
            ), "You must set the number of observations for the test split."
            return len(self._sample_paths) * self._n_combinations
        else:
            return len(self._sample_paths)

    def disable_augs(self) -> None:
        self._augment = False

    def _load_objects_and_grasps(
        self, tiny: bool, split: str, seed: int = 0
    ) -> Tuple[List[Any], List[Any], str]:
        """
        Returns a list of object mesh paths, a list of grasp sequence paths associated, and the dataset name.
        """
        raise NotImplementedError

    def __gettask__(
        self, idx: int, n_context: int, n_target: int
    ) -> Tuple[List[Tuple], List[Tuple]]:
        if self._split == "test":
            assert (
                self._observations_number is not None
            ), "You must set the number of observations for the test split."
            sequence_idx = idx // self._n_combinations
            sample_idx = idx % self._n_combinations
        noisy_grasp_sequence = self._sample_paths[
            idx if not self._split == "test" else sequence_idx
        ]
        assert noisy_grasp_sequence is not None
        assert n_context <= len(noisy_grasp_sequence)
        if self._split == "test":
            # If we're testing/evaluating the model, we want to go through the entire task for
            # ContactPoseDataset and all unique combinations noisy grasp samples for a given number of
            # observations.
            samples_paths = [
                noisy_grasp_sequence[i] for i in self._combinations[sample_idx]
            ]
        else:
            # Randomly sample n_context grasps from the grasp sequence (ignore n_target since we're not
            # doing meta-learning here). We want a random subset of cardinality n_context from the grasp
            # sequence, but they must follow each other in the sequence. So we randomly sample a starting
            # index and then take the next n_context elements.
            start_idx = torch.randint(
                low=0, high=len(noisy_grasp_sequence) - n_context + 1, size=(1,)
            ).item()
            samples_paths = noisy_grasp_sequence[start_idx : start_idx + n_context]
        samples, labels = [], []
        for sample_path in samples_paths:
            with open(sample_path, "rb") as f:
                compressed_pkl = f.read()
                sample, label = pickle.loads(blosc.decompress(compressed_pkl))
            samples.append(sample)
            labels.append(label)
        return torch.stack(samples), torch.stack(labels)
