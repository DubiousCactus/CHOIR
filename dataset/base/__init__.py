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
import random
from typing import Any, List, Optional, Tuple

import blosc2
import numpy as np
import torch
from bps_torch.tools import sample_grid_cube, sample_sphere_uniform
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
        min_views_per_grasp: int,
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
        random_anchor_assignment: bool,
        dataset_name: str,
        use_deltas: bool,
        use_bps_grid: bool,
        load_pointclouds: bool = False,  # For GraspTTA, set to True
        noisy_samples_per_grasp: Optional[int] = None,
    ) -> None:
        # For GRAB, noisy_samples_per_grasp is actually the number of frames in the sequence. At
        # training time it doesn't matter because we sample a random subset of these frames. But at
        # test time we need to go through the entire sequence. By setting noisy_samples_per_grasp
        # to None, we'll figure out the number of frames in the sequence and use that as the number
        # of noisy samples per grasp.
        seq_len = (
            noisy_samples_per_grasp if noisy_samples_per_grasp is not None else 100
        )

        # This feels super hacky because I hadn't designed metabatch to be used this way. It was
        # only meant to be used with the traditional context + target paradigm. This will work but
        # requires min_pts=1 and max_ctx_pts=1, as well as using n_target. Don't change any of
        # these defaults or it will break for this case.
        seq_len = 1 if seq_len == 0 else seq_len  # TODO: wtf is this??
        assert seq_len >= max_views_per_grasp
        super().__init__(
            min_pts=min_views_per_grasp,
            max_ctx_pts=max_views_per_grasp,
            max_tgt_pts=max_views_per_grasp
            + 1,  # This is actually irrelevant in our case! Just needs to be higher than max_ctx_pts
            total_tgt_pts=seq_len,
            eval=False,
            predict_full_target=False,
            predict_full_target_during_eval=False,
        )
        self._load_pointclouds = load_pointclouds
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
        self._seq_len = noisy_samples_per_grasp
        self._random_anchor_assignment = random_anchor_assignment
        self._seq_lengths = []  # For variable length sequences (GRAB)
        self._dataset_name = dataset_name
        self._use_bps_grid = use_bps_grid
        # self._perturbations = [] # TODO: Implement
        self._cache_dir = osp.join(
            get_original_cwd(), "data", f"{dataset_name}_preprocessed"
        )
        self._eval_mode = False
        if not osp.isdir(self._cache_dir):
            os.makedirs(self._cache_dir)
        grid_len = None
        if use_bps_grid:
            grid_len = int(round(self._bps_dim ** (1 / 3)))
            assert (
                grid_len % 2 == 0
            ), f"Grid length must be even. Got {grid_len} for bps_dim={self._bps_dim}."

        bps_path = osp.join(
            get_original_cwd(),
            "data",
            f"bps_{f'{grid_len}-grid' if use_bps_grid else self._bps_dim}_{rescale}-rescaled.pkl",
        )
        anchor_indices_path = osp.join(
            get_original_cwd(),
            "data",
            f"anchor_indices_{self._bps_dim}_{'random' if random_anchor_assignment else 'ordered'}.pkl",
        )
        if osp.isfile(bps_path):
            with open(bps_path, "rb") as f:
                bps = pickle.load(f)
        else:
            if use_bps_grid:
                bps = sample_grid_cube(
                    grid_size=grid_len,
                    n_dims=3,
                    minv=-0.1 if rescale == "none" else -0.6,
                    maxv=0.1 if rescale == "none" else 0.6,
                ).cpu()
            else:
                bps = sample_sphere_uniform(
                    n_points=self._bps_dim,
                    n_dims=3,
                    radius=0.2 if rescale == "none" else 0.6,
                    random_seed=1995,
                ).cpu()
            with open(bps_path, "wb") as f:
                pickle.dump(bps, f)
        if osp.isfile(anchor_indices_path):
            with open(anchor_indices_path, "rb") as f:
                anchor_indices = pickle.load(f)
        else:
            anchor_indices = (
                torch.arange(0, 32).repeat((self._bps_dim // 32,)).cpu().numpy()
            )
            if random_anchor_assignment:
                random.Random(1995).shuffle(anchor_indices)
            with open(anchor_indices_path, "wb") as f:
                pickle.dump(anchor_indices, f)
        self._bps = bps
        self._anchor_indices = torch.from_numpy(anchor_indices).type(torch.int64)
        self._center_on_object_com = center_on_object_com
        self._rescale = rescale
        self._remap_bps_distances = remap_bps_distances
        self._exponential_map_w = exponential_map_w
        self._use_deltas = use_deltas
        self._observations_number = None
        self._n_combinations = None
        self._combinations = None
        self._debug = debug
        (
            objects_w_contacts,
            grasps,
            dataset_name,
        ) = self._load_objects_and_grasps(tiny, split, seed=seed)
        assert len(objects_w_contacts) == len(grasps)
        assert len(grasps) > 0
        self._sample_paths: List[List[str]] = self._load(
            split, objects_w_contacts, grasps, dataset_name
        )
        assert len(self._sample_paths) > 0

    @property
    def use_bps_grid(self):
        return self._use_bps_grid

    @property
    def use_deltas(self):
        return self._use_deltas

    @property
    def eval_anchor_assignment(self):
        return False

    @property
    def eval_observation_plateau(self):
        return False

    @property
    def theta_dim(self):
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self._dataset_name.lower()

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

    @property
    def anchor_indices(self) -> torch.Tensor:
        return self._anchor_indices

    @property
    def center_on_object_com(self) -> bool:
        return self._center_on_object_com

    def set_observations_number(self, n: int) -> None:
        # This SO thread explains the problem and the (super simple) solution:
        # https://stackoverflow.com/questions/27974126/get-all-n-choose-k-combinations-of-length-n
        self._observations_number = n
        if self._seq_len is not None:
            if self.eval_observation_plateau:
                # We don't wanna go through all combinations since that takes forever on more than
                # 5 observations :( Instead, we'll randomly sample 100 combinations and use those
                # for the evaluation. This should approximate the true accuracy anyways, and we're
                # not interested in very precise numbers here.
                comb = list(itertools.combinations(range(self._seq_len), n))
                self._combinations = (
                    random.sample(comb, 100) if len(comb) > 100 else comb
                )
            else:
                self._combinations = list(
                    itertools.combinations(range(self._seq_len), n)
                )
            self._n_combinations = len(self._combinations)
        else:
            # We need to do things differently for GRAB, where we have sequences of variable
            # length. In each sequence, we want to sample N observations but still predict for
            # each frame in the sequence. The N observations include that frame, so for the first
            # frame we need to samples N times frame 0, and N-1 times frame 1, etc.
            self._n_combinations = None
            self._seq_lengths = [len(seq) for seq in self._sample_paths]

    def _mine_neg_pos_pairs(self):
        """
        Mine negative-positive pairs for the training split using the Procrustes distance.
        """
        raise NotImplementedError

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
            if self._seq_len is not None:
                return len(self._sample_paths) * self._n_combinations
            else:
                # We need to do things differently for GRAB, where we have sequences of variable
                # length. In each sequence, we want to sample N observations but still predict for
                # each frame in the sequence. The N observations include that frame, so for the first
                # frame we need to samples N times frame 0, and N-1 times frame 1, etc.
                return sum([len(seq) for seq in self._sample_paths])
        else:
            return len(self._sample_paths)

    def set_eval_mode(self, eval: bool) -> None:
        self._eval_mode = eval

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
        if self._split == "test" and self._seq_len is not None:
            assert (
                self._observations_number is not None
            ), "You must set the number of observations for the test split."
            sequence_idx = idx // self._n_combinations
            sample_idx = idx % self._n_combinations
        elif self._split == "test" and self._seq_len is None:
            assert (
                self._observations_number is not None
            ), "You must set the number of observations for the test split."
            # This is for GRAB when we need to do this differently for each sequence because they
            # are of variable length.
            acc = 0
            for i, seq_len in enumerate(self._seq_lengths):
                if idx < (seq_len + acc):
                    sequence_idx = i
                    break
                acc += seq_len
            sample_idx = idx - acc

        noisy_grasp_sequence = self._sample_paths[
            idx if not self._split == "test" else sequence_idx
        ]
        assert noisy_grasp_sequence is not None
        # n_context = min(len(noisy_grasp_sequence), n_context) # Can't do that because other
        # workers won't be aware of len(noisy_grasp_sequence) since each batch element is a
        # different grasp sequence.
        assert n_context <= len(noisy_grasp_sequence)
        if self._split == "test":
            if self._combinations is not None:  # ContactPose
                # If we're testing/evaluating the model, we want to go through the entire task for
                # ContactPoseDataset and all unique combinations noisy grasp samples for a given number of
                # observations.
                samples_paths = [
                    noisy_grasp_sequence[i] for i in self._combinations[sample_idx]
                ]
            else:
                # GRAB (variable sequence length, going frame by frame with previous
                # self._observations_number frames in the context window
                # Sample self._observations_number frames which are preceiding the current frame
                # (sample_idx) in the sequence sequence_idx.
                samples_paths = [
                    noisy_grasp_sequence[
                        max(0, sample_idx - self._observations_number + i + 1)
                    ]
                    for i in range(self._observations_number)
                ]
        else:  # Training / validation
            # Randomly sample n_context grasps from the grasp sequence (ignore n_target since we're not
            # doing meta-learning here). We want a random subset of cardinality n_context from the grasp
            # sequence, but they must follow each other in the sequence. So we randomly sample a starting
            # index and then take the next n_context elements.
            start_idx = torch.randint(
                low=0, high=len(noisy_grasp_sequence) - n_context + 1, size=(1,)
            ).item()
            samples_paths = noisy_grasp_sequence[start_idx : start_idx + n_context]

        (
            choir,
            rescaled_ref_pts,
            scalar,
            hand_idx,
            theta,
            beta,
            rot_6d,
            trans,
            joints,
            anchors,
            gt_choir,
            gt_rescaled_ref_pts,
            gt_scalar,
            gt_joints,
            gt_anchors,
            gt_theta,
            gt_beta,
            gt_rot_6d,
            gt_trans,
            gt_contact_gaussians,
            mesh_pths,
            gt_obj_pts,
            gt_obj_contacts,
        ) = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )

        for sample_path in samples_paths:
            success = False
            failures = 0
            while not success:
                try:
                    with open(sample_path, "rb") as f:
                        compressed_pkl = f.read()
                        sample, label, mesh_pth = pickle.loads(
                            blosc2.decompress(compressed_pkl)
                        )
                except RuntimeError as e:
                    if failures == 5:
                        os.remove(sample_path)
                        raise e
                    print(
                        f"Couldn't decompress {sample_path}! Trying again ({failures}/5)"
                    )
                    failures += 1
                else:
                    success = True
            choir.append(sample[0])
            gt_choir.append(label[0])
            if self._dataset_name.lower() == "contactpose" and self._load_pointclouds:
                gt_obj_pts.append(label[-2])
                gt_obj_contacts.append(label[-1])
            elif self._dataset_name.lower() == "contactpose":
                gt_obj_pts.append(np.zeros(1))
                gt_obj_contacts.append(np.zeros(1))
            theta.append(sample[4])
            beta.append(sample[5])
            rot_6d.append(sample[6])
            trans.append(sample[7])
            gt_theta.append(label[5])
            gt_beta.append(label[6])
            gt_rot_6d.append(label[7])
            gt_trans.append(label[8])
            mesh_pths.append(mesh_pth)
            if self._eval_mode:
                rescaled_ref_pts.append(sample[1])
                scalar.append(sample[2])
                hand_idx.append(sample[3])
                if len(sample) > 8:
                    # To avoid recomputing the dataset for experiments that don't need sample
                    # joints/anchors
                    joints.append(sample[8])
                    anchors.append(sample[9])
                gt_rescaled_ref_pts.append(label[1])
                gt_scalar.append(label[2])
                gt_joints.append(label[3])
                gt_anchors.append(label[4])
                gt_contact_gaussians.append(label[9] if len(label) > 9 else np.zeros(1))

        if self._eval_mode:
            # Some people claim that torch.from_numpy is faster than torch.stack in most cases...
            sample = {
                "choir": torch.from_numpy(np.array([a for a in choir])),
                "rescaled_ref_pts": torch.from_numpy(
                    np.array([a for a in rescaled_ref_pts])
                ),
                "scalar": torch.from_numpy(np.array([a.squeeze() for a in scalar])),
                "is_rhand": torch.from_numpy(np.array([a for a in hand_idx])),
                "theta": torch.from_numpy(np.array([a for a in theta])),
                "beta": torch.from_numpy(np.array([a for a in beta])),
                "rot": torch.from_numpy(np.array([a for a in rot_6d])),
                "trans": torch.from_numpy(np.array([a for a in trans])),
                "joints": torch.from_numpy(np.array([a for a in joints])),
                "anchors": torch.from_numpy(np.array([a for a in anchors])),
            }
            label = {
                "choir": torch.from_numpy(np.array([a for a in gt_choir])),
                "rescaled_ref_pts": torch.from_numpy(
                    np.array([a for a in gt_rescaled_ref_pts])
                ),
                "scalar": torch.from_numpy(np.array([a.squeeze() for a in gt_scalar])),
                "joints": torch.from_numpy(np.array([a for a in gt_joints])),
                "anchors": torch.from_numpy(np.array([a for a in gt_anchors])),
                "theta": torch.from_numpy(np.array([a for a in gt_theta])),
                "beta": torch.from_numpy(np.array([a for a in gt_beta])),
                "rot": torch.from_numpy(np.array([a for a in gt_rot_6d])),
                "trans": torch.from_numpy(np.array([a for a in gt_trans])),
                "contact_gaussians": torch.from_numpy(
                    np.array([a for a in gt_contact_gaussians])
                ),
            }
        else:
            # This will hopefully save a lot of memory and allow to bump up the batch size during
            # training :) :)
            # In retrospect: it didn't. The whole project has mutated several times into its final
            # ugly form.
            sample = {
                "choir": torch.from_numpy(np.array([a for a in choir])),
                "theta": torch.from_numpy(np.array([a for a in theta])),
                "beta": torch.from_numpy(np.array([a for a in beta])),
                "rot": torch.from_numpy(np.array([a for a in rot_6d])),
                "trans": torch.from_numpy(np.array([a for a in trans])),
            }
            label = {
                "choir": torch.from_numpy(np.array([a for a in gt_choir])),
                "theta": torch.from_numpy(np.array([a for a in gt_theta])),
                "beta": torch.from_numpy(np.array([a for a in gt_beta])),
                "rot": torch.from_numpy(np.array([a for a in gt_rot_6d])),
                "trans": torch.from_numpy(np.array([a for a in gt_trans])),
            }

        if self._dataset_name.lower() == "contactpose":
            label["obj_pts"] = torch.from_numpy(
                np.array([a for a in gt_obj_pts])
            ).float()
            label["obj_contacts"] = torch.from_numpy(
                np.array([a for a in gt_obj_contacts])
            ).float()

        else:
            label["obj_pts"] = torch.tensor(0.0)
            label["obj_contacts"] = torch.tensor(0.0)
        return sample, label, mesh_pths
