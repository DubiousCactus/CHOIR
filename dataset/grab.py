#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
GRAB dataset customized for the project.
"""


import hashlib
import os
import os.path as osp
import pickle
import random
from contextlib import redirect_stdout
from copy import deepcopy
from typing import Any, List, Tuple

import blosc
import numpy as np
import smplx
import torch
from open3d import io as o3dio
from pytorch3d.transforms import Transform3d, axis_angle_to_matrix
from torchmetrics import MeanMetric
from tqdm import tqdm
from trimesh import Trimesh

from conf.project import ANSI_COLORS, Theme
from model.affine_mano import AffineMANO
from utils import colorize, to_cuda_
from utils.dataset import compute_choir, get_scalar, pack_and_pad_sample_label
from utils.visualization import (
    visualize_CHOIR,
    visualize_CHOIR_prediction,
    visualize_MANO,
)

from .base import BaseDataset


class GRABDataset(BaseDataset):
    base_unit = 1000.0  # The dataset is in meters, we want to work in mm.

    def __init__(
        self,
        root_path: str,
        smplx_path: str,
        split: str,
        validation_objects: int = 3,
        test_objects: int = 5,
        perturbation_level: int = 0,
        obj_ptcld_size: int = 3000,
        bps_dim: int = 1024,
        right_hand_only: bool = True,
        center_on_object_com: bool = True,
        max_views_per_grasp: int = 5,  # Corresponds to window frames here
        tiny: bool = False,
        augment: bool = False,
        n_augs: int = 1,
        seed: int = 0,
        debug: bool = False,
        rescale: str = "none",
        remap_bps_distances: bool = False,
        exponential_map_w: float = 5.0,
    ) -> None:
        self._perturbations = [
            {"trans": 0.0, "rot": 0.0, "pca": 0.0},  # Level 0
            {"trans": 0.02, "rot": 0.05, "pca": 0.3},  # Level 1
            {
                "trans": 0.05,
                "rot": 0.15,
                "pca": 0.5,
            },  # Level 2 (0.05m, 0.15rad, 0.5 PCA units)
        ]
        self._root_path = root_path
        self._smplx_path = smplx_path
        super().__init__(
            dataset_name="GRAB",
            bps_dim=bps_dim,
            validation_objects=validation_objects,
            test_objects=test_objects,
            right_hand_only=right_hand_only,
            obj_ptcld_size=obj_ptcld_size,
            perturbation_level=perturbation_level,
            max_views_per_grasp=max_views_per_grasp,
            # noisy_samples_per_grasp is just indicative for __len__() but it doesn't matter much
            # since we'll sample frame windows on the fly.
            noisy_samples_per_grasp=1000,
            rescale=rescale,
            center_on_object_com=center_on_object_com,
            remap_bps_distances=remap_bps_distances,
            exponential_map_w=exponential_map_w,
            augment=augment,
            n_augs=n_augs,
            split=split,
            tiny=tiny,
            seed=seed,
            debug=debug,
        )

    def _load(
        self,
        split: str,
        objects: List,
        grasp_sequences: List,
        dataset_name: str,
    ) -> List[List[str]]:
        """
        Returns a list of noisy grasp sequences.
        """
        samples_labels_pickle_pth = osp.join(
            self._cache_dir,
            "samples_and_labels",
            f"dataset_{hashlib.shake_256(dataset_name.encode()).hexdigest(8)}_"
            + f"perturbed-{self._perturbation_level}_"
            + f"{self._bps_dim}-bps_"
            + f"{'object-centered_' if self._center_on_object_com else ''}"
            + f"{self._rescale}_rescaled_"
            + f"{'exponential_mapped' if self._remap_bps_distances else ''}"
            + (f"{self._exponential_map_w}_" if self._remap_bps_distances else "")
            + f"{split}",
        )
        if not osp.isdir(samples_labels_pickle_pth):
            os.makedirs(samples_labels_pickle_pth)
        affine_mano: AffineMANO = to_cuda_(AffineMANO(ncomps=21, flat_hand_mean=True))  # type: ignore
        choir_paths = []
        print("[*] Computing CHOIR fields...")
        dataset_mpjpe = MeanMetric()
        for mesh_pth, grasp_seq_pth_w_grasping_hand in tqdm(
            zip(objects, grasp_sequences), total=len(objects)
        ):
            grasp_seq_pth, grasping_hand = grasp_seq_pth_w_grasping_hand
            seq = np.load(grasp_seq_pth, allow_pickle=True)
            seq = {k: seq[k].item() for k in seq.files}
            """
            {
                'gender': str,
                'sbj_id': str, # 's1' to 's10'
                'obj_name': str,
                'motion_intent': str,
                'framerate': float, # 120FPS for all
                'n_frames': int,
                'n_comps': int, # PCA components for SMPL-X
                'body': {'params': Dict, 'vtemp': str}, # SMPL-X params and path to shape template
                'lhand': {
                    'params': {
                        'global_orient': np.ndarray,
                        'hand_pose': np.ndarray,
                        'transl': np.ndarray,
                        'fullpose': np.ndarray,
                    }, # global translation and joint rotation (axis-angle). But in fact there's more?
                    'vtemp': str # relative path to personalized shape (very annoying though, why not use beta??)
                },
                'rhand': {...},
                'object': {
                    'params': {
                        'global_orient': np.ndarray,
                        'transl': np.ndarray,
                        'contact': int, # binary?
                    }, # global translation and rotation
                    'object_mesh': str, # relative path to object mesh
                },
                'table': {}, # Not used
                'contact': int, # Index to the body part in contact with the object
            }
            """
            obj_name = seq["obj_name"]
            intent = seq["motion_intent"]
            p_num = seq["sbj_id"]
            grasp_name = f"{obj_name}_{p_num}_{intent}"
            if not osp.isdir(osp.join(samples_labels_pickle_pth, grasp_name)):
                os.makedirs(osp.join(samples_labels_pickle_pth, grasp_name))
            if len(os.listdir(osp.join(samples_labels_pickle_pth, grasp_name))) > 0:
                choir_paths.append(
                    [
                        os.path.join(samples_labels_pickle_pth, grasp_name, f)
                        for f in sorted(
                            os.listdir(osp.join(samples_labels_pickle_pth, grasp_name))
                        )
                    ]
                )
            else:
                obj_mesh = o3dio.read_triangle_mesh(mesh_pth)
                obj_ptcld = torch.from_numpy(
                    np.asarray(
                        obj_mesh.sample_points_uniformly(self._obj_ptcld_size).points  # type: ignore
                    )
                )
                visualize = self._debug and (random.Random().random() < 0.05)
                has_visualized = False
                # ================== Original Hand-Object Pair ==================
                # TODO: Load betas from file and try using AffineMANO
                h_mesh = os.path.join(self._root_path, seq[grasping_hand]["vtemp"])
                h_vtemp = np.array(
                    o3dio.read_triangle_mesh(h_mesh).vertices
                )  # Or with Trimesh
                gt_params = {
                    k: torch.from_numpy(v).type(torch.float32)
                    for k, v in seq[grasping_hand]["params"].items()
                }
                with torch.no_grad(), redirect_stdout(None):
                    h_m = to_cuda_(
                        smplx.create(
                            model_path=self._smplx_path,
                            model_type="mano",
                            is_rhand=grasping_hand == "rhand",
                            v_template=h_vtemp,
                            num_pca_comps=seq["n_comps"],
                            flat_hand_mean=True,
                            batch_size=seq["n_frames"],
                        )
                    )
                    mano_result = h_m(**to_cuda_(gt_params))
                gt_verts, gt_joints = mano_result.vertices, mano_result.joints
                # Now I must center the grasp on the object s.t. the object is at the origin
                # and in canonical pose. This is done by applying the inverse of the object's
                # global orientation and translation to the hand's global orientation and
                # translation. We can use pytorch3d's transform_points for this.
                obj_transform = (
                    Transform3d()
                    .rotate(
                        axis_angle_to_matrix(
                            torch.from_numpy(seq["object"]["params"]["global_orient"])
                        )
                    )
                    .translate(torch.from_numpy(seq["object"]["params"]["transl"]))
                )
                obj_transform = obj_transform.inverse().cuda()
                gt_verts = obj_transform.transform_points(gt_verts)
                gt_joints = obj_transform.transform_points(gt_joints)
                # =================== Apply augmentation =========================
                # TODO: Get gt_hTm first (if I manage to use AffineMANO)
                # if self._augment and k > 0:
                # obj_mesh, gt_hTm = augment_hand_object_pose(
                # obj_mesh, gt_hTm, around_z=False
                # )
                # =================================================================
                # ============ Shift the pair to the object's center ============
                if self._center_on_object_com:
                    # TODO: if self._center_on_object_com and not (self._augment and k > 0):
                    obj_center = torch.from_numpy(obj_mesh.get_center())
                    obj_mesh.translate(-obj_center)
                    obj_ptcld -= obj_center.to(obj_ptcld.device)
                    gt_verts -= obj_center.to(gt_verts.device)
                    gt_joints -= obj_center.to(gt_joints.device)
                # ================================================================
                gt_anchors = affine_mano.get_anchors(gt_verts)
                # ================== Rescaled Hand-Object Pair ==================
                gt_scalar = get_scalar(gt_anchors, obj_ptcld, self._rescale)

                # I know it's bad to do CUDA stuff in the dataset if I want to use multiple
                # workers, but bps_torch is forcing my hand here so I might as well help it.
                if self._debug:
                    import timeit

                    start = timeit.default_timer()
                gt_choir, gt_rescaled_ref_pts = compute_choir(
                    to_cuda_(obj_ptcld).unsqueeze(0),
                    to_cuda_(gt_anchors),
                    scalar=gt_scalar,
                    bps=to_cuda_(self._bps).unsqueeze(0),  # type: ignore
                    remap_bps_distances=self._remap_bps_distances,
                    exponential_map_w=self._exponential_map_w,
                )
                if self._debug:
                    end = timeit.default_timer()
                    print(
                        f"CHOIR computation time for N={seq['n_frames']}: {(end - start)*1000:.3f}ms"
                    )

                choir_sequence_paths = []
                # ================== Perturbed Hand-Object Pair ==================
                for i in range(seq["n_frames"]):
                    sample_pth = osp.join(
                        samples_labels_pickle_pth, grasp_name, f"sample_{i:06d}.pkl"
                    )
                    if osp.isfile(sample_pth):
                        choir_sequence_paths.append(sample_pth)
                        continue
                    # Distance hand-object, with object at origin:
                    dist = torch.norm(torch.mean(gt_verts[i], dim=0))
                    # We'll use 20FPS so we'll skip every 6 frames
                    # Filter the sequence frames to keep only those where the hand is within 50cm of the object
                    if i % 6 != 0 or dist > 0.5:
                        continue

                    h_params = {
                        k: torch.from_numpy(v[None, i]).type(torch.float32)
                        for k, v in deepcopy(seq[grasping_hand]["params"]).items()
                    }

                    trans_noise = (
                        torch.randn(3, device=h_params["transl"].device)
                        * self._perturbations[self._perturbation_level]["trans"]
                    )
                    pose_noise = torch.cat(
                        [
                            torch.randn(24, device=h_params["hand_pose"].device)
                            * self._perturbations[self._perturbation_level]["pca"],
                        ]
                    )
                    global_orient_noise = (
                        torch.randn(3, device=h_params["global_orient"].device)
                        * self._perturbations[self._perturbation_level]["rot"]
                    )
                    h_params["hand_pose"] += pose_noise
                    h_params["transl"] += trans_noise
                    h_params["global_orient"] += global_orient_noise

                    with torch.no_grad(), redirect_stdout(None):
                        h_m = to_cuda_(
                            smplx.create(
                                model_path=self._smplx_path,
                                model_type="mano",
                                is_rhand=grasping_hand == "rhand",
                                v_template=h_vtemp,
                                num_pca_comps=seq["n_comps"],
                                flat_hand_mean=True,
                                batch_size=1,
                            )
                        )
                        mano_result = h_m(**to_cuda_(h_params))
                    verts, joints = mano_result.vertices, mano_result.joints
                    anchors = affine_mano.get_anchors(verts)
                    # Again, bring back to object's coordinate system:
                    verts = obj_transform[i].transform_points(verts)
                    joints = obj_transform[i].transform_points(joints)
                    anchors = obj_transform[i].transform_points(anchors)
                    scalar = get_scalar(anchors, obj_ptcld, self._rescale)

                    mpjpe = torch.linalg.vector_norm(
                        joints.squeeze(0) - gt_joints.squeeze(0), dim=1, ord=2
                    ).mean()
                    dataset_mpjpe.update(mpjpe.item())
                    # I know it's bad to do CUDA stuff in the dataset if I want to use multiple
                    # workers, but bps_torch is forcing my hand here so I might as well help it.
                    choir, rescaled_ref_pts = compute_choir(
                        to_cuda_(obj_ptcld).unsqueeze(0),
                        to_cuda_(anchors),
                        scalar=scalar,
                        bps=to_cuda_(self._bps).unsqueeze(0),  # type: ignore
                        remap_bps_distances=self._remap_bps_distances,
                        exponential_map_w=self._exponential_map_w,
                    )
                    sample, label = pack_and_pad_sample_label(
                        choir,
                        rescaled_ref_pts,
                        scalar,
                        torch.ones((1, 1)).cuda()
                        if grasping_hand == "rhand"
                        else torch.zeros((1, 1)).cuda(),  # (1, 1)
                        gt_choir,
                        gt_rescaled_ref_pts,
                        gt_scalar,
                        gt_joints,
                        gt_anchors,
                        torch.zeros((1, 21)).cuda(),  # (1, 18)
                        torch.zeros((1, 10)).cuda(),  # (1, 10)
                        torch.zeros((1, 6)).cuda(),  # (1, 6)
                        torch.zeros((1, 3)).cuda(),  # (1, 3)
                        self._bps_dim,
                    )
                    # =================================================================

                    with open(sample_pth, "wb") as f:
                        pkl = pickle.dumps((sample, label))
                        compressed_pkl = blosc.compress(pkl)
                        f.write(compressed_pkl)
                    choir_sequence_paths.append(sample_pth)
                    if (
                        visualize
                        and not has_visualized
                        and random.Random().random() < 0.01
                    ):
                        print("[*] Plotting CHOIR... (please be patient)")
                        visualize_CHOIR(
                            gt_choir[i],
                            self._bps,
                            gt_scalar,
                            gt_verts[i],
                            gt_anchors[i],
                            obj_mesh,
                            obj_ptcld,
                            gt_rescaled_ref_pts.squeeze(0),
                            affine_mano,
                        )
                        faces = h_m.faces
                        gt_MANO_mesh = Trimesh(gt_verts[i].cpu().numpy(), faces)
                        pred_MANO_mesh = Trimesh(verts.squeeze(0).cpu().numpy(), faces)
                        visualize_MANO(
                            pred_MANO_mesh, obj_mesh=obj_mesh, gt_hand=gt_MANO_mesh
                        )
                        visualize_CHOIR_prediction(
                            gt_choir[i].unsqueeze(0),
                            gt_choir[i].unsqueeze(0),
                            self._bps,
                            scalar,
                            gt_scalar,
                            rescaled_ref_pts,
                            gt_rescaled_ref_pts,
                            gt_verts[i].unsqueeze(0),
                            gt_joints[i].unsqueeze(0),
                            gt_anchors[i].unsqueeze(0),
                            is_rhand=(grasping_hand == "rhand"),
                            use_smplx=True,
                            remap_bps_distances=self._remap_bps_distances,
                            exponential_map_w=self._exponential_map_w,
                        )
                        has_visualized = True
                choir_paths.append(choir_sequence_paths)
        print(
            f"[*] Dataset MPJPE (mm): {dataset_mpjpe.compute().item() * self.base_unit}"
        )
        return choir_paths

    def _load_objects_and_grasps(
        self, tiny: bool, split: str, seed: int = 0
    ) -> Tuple[List[Any], List[Any], str]:
        """
        Returns a list of object mesh paths, a list of grasp sequence paths associated, and the dataset name.
        """
        objects, grasp_sequences = [], []
        n_participants = 10 if not tiny else 3
        # 1. Get list of objects so we can split them into train/val/test. The object name is
        # <obj_name>_*.npz in the object path.
        object_names = []
        assert os.path.isdir(
            self._root_path
        ), f"root_path {self._root_path} is not a directory"
        for _, _, files in os.walk(self._root_path):
            object_names += [f.split("_")[0] for f in files if f.endswith(".npz")]
        object_names = sorted(list(set(object_names)))
        random.Random(seed).shuffle(object_names)
        assert (self._validation_objects + self._test_objects) < len(object_names), (
            f"validation_objects + test_objects ({self._validation_objects} + {self._test_objects})"
            + f" must be less than n_objects ({len(object_names)})"
        )
        assert (
            self._validation_objects >= 1
        ), f"validation_objects ({self._validation_objects}) must be greater or equal to 1"
        assert (
            self._test_objects >= 1
        ), f"test_objects ({self._test_objects}) must be greater or equal to 1"
        if split == "train":
            object_names = object_names[
                : -(self._validation_objects + self._test_objects)
            ]
        elif split == "val":
            object_names = object_names[
                -(self._validation_objects + self._test_objects) : -self._test_objects
            ]
        elif split == "test":
            object_names = object_names[-self._test_objects :]

        """ Alternatively, we may use the official splits:
        {
            'test': ['mug', 'wineglass', 'camera', 'binoculars', 'fryingpan', 'toothpaste'],
            'val': ['apple', 'toothbrush', 'elephant', 'hand'],
            'train': [] # All others
        }
        """

        # 2. Load the object paths and grasp sequences paths
        print(f"[*] Loading GRAB{' (tiny)' if tiny else ''}...")
        dataset_path = osp.join(
            self._cache_dir,
            f"dataset_{split}_{n_participants}-participants"
            + f"_{self._validation_objects}-val-held-out"
            + f"_{self._test_objects}-test-held-out"
            + f"_{self._obj_ptcld_size}-obj-pts"
            + f"{'right-hand' if self._right_hand_only else 'both-hands'}_seed-{seed}.pkl",
        )
        if os.path.isfile(dataset_path):
            with open(dataset_path, "rb") as f:
                compressed_pkl = f.read()
                objects, grasp_sequences, n_left, n_right = pickle.loads(
                    blosc.decompress(compressed_pkl)
                )
        else:
            n_left, n_right = 0, 0
            pbar = tqdm(total=n_participants)
            n = 0
            for root, dirs, files in os.walk(self._root_path):
                if os.path.basename(root) in ["tools", "SMPLX"]:
                    del dirs[:]
                    continue
                for f in files:
                    if f.endswith(".npz"):
                        obj_name = f.split("_")[0]
                        if obj_name in object_names:
                            seq = np.load(os.path.join(root, f), allow_pickle=True)
                            seq = {k: seq[k].item() for k in seq.files}
                            object_path = os.path.join(
                                self._root_path, seq["object"]["object_mesh"]
                            )  # PLY files
                            assert os.path.isfile(
                                object_path
                            ), f"object_path {object_path} is not a file"
                            # Contacts are represented as a body part index. The right hand is between
                            # 41 and 55 and all objects must be grasped by *a* hand, so those falling
                            # outside of this range are grasped by the left hand. See
                            # https://github.com/otaheri/GRAB/blob/4dab3211fae4fc5b8eb6ab86246ccc3a42d8f611/tools/utils.py#L166
                            # TODO: What happens in handover sequences? Let's ignore them for now.
                            # In the future we'll have to carefully sample time windows! (or not?)
                            is_right_hand = (seq["contact"]["object"] >= 41).any()
                            is_left_hand = (
                                seq["contact"]["object"][
                                    (seq["contact"]["object"] >= 26)
                                ]
                                < 41
                            ).any()
                            is_handover = is_right_hand and is_left_hand
                            if (
                                self._right_hand_only and not is_right_hand
                            ) or is_handover:
                                continue
                            n_right += 1 if is_right_hand else 0
                            n_left += 1 if not is_right_hand else 0
                            grasp_sequences.append(
                                (
                                    os.path.join(root, f),
                                    "rhand" if is_right_hand else "lhand",
                                )
                            )
                            objects.append(object_path)
                n += 1
                pbar.update()
                if n == n_participants:
                    break
            with open(dataset_path, "wb") as f:
                compressed_pkl = blosc.compress(
                    pickle.dumps((objects, grasp_sequences, n_left, n_right))
                )
                f.write(compressed_pkl)
        print(
            f"[*] Loaded {len(object_names)} objects and {len(grasp_sequences)} grasp sequences ({n_left} left hand, {n_right} right hand)"
        )
        print(
            colorize(
                f"[*] {'Training' if split == 'train' else 'Validation'} objects: {', '.join(object_names)}",
                ANSI_COLORS[
                    Theme.TRAINING.value if split == "train" else Theme.VALIDATION.value
                ],
            )
        )
        assert len(objects) == len(grasp_sequences)
        dataset_name = os.path.basename(dataset_path).split(".")[0]
        return objects, grasp_sequences, dataset_name
