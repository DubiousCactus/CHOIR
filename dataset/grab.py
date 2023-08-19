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
from typing import Any, Dict, List, Tuple

import blosc
import numpy as np
import smplx
import torch
from open3d import io as o3dio
from pytorch3d.transforms import (
    Transform3d,
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_rotation_6d,
)
from torchmetrics import MeanMetric
from tqdm import tqdm
from trimesh import Trimesh

from conf.project import ANSI_COLORS, Theme
from model.affine_mano import AffineMANO
from utils import colorize, to_cuda, to_cuda_
from utils.dataset import compute_choir, get_scalar, pack_and_pad_sample_label
from utils.visualization import ScenePicAnim, visualize_MANO

from .base import BaseDataset


class GRABDataset(BaseDataset):
    base_unit = 1000.0  # The dataset is in meters, we want to work in mm.

    def __init__(
        self,
        root_path: str,
        smplx_path: str,
        split: str,
        static_grasps_only: bool = False,
        validation_objects: int = 3,
        test_objects: int = 5,
        perturbation_level: int = 0,
        obj_ptcld_size: int = 3000,
        bps_dim: int = 1024,
        right_hand_only: bool = True,
        center_on_object_com: bool = True,
        min_views_per_grasp: int = 2,
        max_views_per_grasp: int = 5,  # Corresponds to window frames here
        tiny: bool = False,
        augment: bool = False,
        n_augs: int = 1,
        seed: int = 0,
        debug: bool = False,
        rescale: str = "none",
        remap_bps_distances: bool = False,
        exponential_map_w: float = 5.0,
        use_affine_mano: bool = False,
        use_official_splits: bool = True,
    ) -> None:
        self._perturbations = [
            {"trans": 0.0, "rot": 0.0, "pca": 0.0},  # Level 0
            {"trans": 0.01, "rot": 0.05, "pca": 0.3},  # Level 1
            {
                "trans": 0.05,
                "rot": 0.15,
                "pca": 0.5,
            },  # Level 2 (0.05m, 0.15rad, 0.5 PCA units)
        ]
        self._root_path = root_path
        self._smplx_path = smplx_path
        self._use_affine_mano = use_affine_mano
        base_betas_path = osp.join(self._root_path, "tools", "subject_meshes")
        self._beta_paths = {
            **{
                f: osp.join(base_betas_path, "male", f)
                for f in os.listdir(osp.join(base_betas_path, "male"))
                if f.endswith("hand_betas.npy")
            },
            **{
                f: osp.join(base_betas_path, "female", f)
                for f in os.listdir(osp.join(base_betas_path, "female"))
                if f.endswith("hand_betas.npy")
            },
        }
        self._affine_mano: AffineMANO = to_cuda_(
            AffineMANO(ncomps=24, flat_hand_mean=True, for_contactpose=False)
        )  # type: ignore
        self._use_official_splits = use_official_splits
        self._static_grasps_only = static_grasps_only
        super().__init__(
            dataset_name="GRAB",
            bps_dim=bps_dim,
            validation_objects=validation_objects,
            test_objects=test_objects,
            right_hand_only=right_hand_only,
            obj_ptcld_size=obj_ptcld_size,
            perturbation_level=perturbation_level,
            min_views_per_grasp=min_views_per_grasp,
            max_views_per_grasp=max_views_per_grasp,
            # noisy_samples_per_grasp is just indicative for __len__() but it doesn't matter much
            # since we'll sample frame windows on the fly.
            noisy_samples_per_grasp=None,
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

    @property
    def theta_dim(self):
        return 24 + (3 if self._use_affine_mano else 0)

    def _mask_out_sequence(
        self, seq: Dict[str, np.ndarray], mask: np.ndarray
    ) -> Dict[str, np.ndarray]:
        # Apply the frame mask to the sequence (skip the body parameters and everything we
        # don't need):
        seq_copy = {k: deepcopy(v) for k, v in seq.items()}
        seq_copy["n_frames"] = mask.sum()
        indices = np.where(mask)[0]
        for hand in ["lhand", "rhand"]:
            for params in ["global_orient", "hand_pose", "transl"]:
                seq_copy[hand]["params"][params] = seq_copy[hand]["params"][params][
                    indices
                ]
        for params in ["global_orient", "transl"]:
            seq_copy["object"]["params"][params] = seq_copy["object"]["params"][params][
                indices
            ]
        return seq_copy

    @to_cuda
    def _load_sequence_params(
        self, seq, p_num: str, grasping_hand: str
    ) -> Dict[str, torch.Tensor]:
        n_frames = seq["n_frames"]
        params = {
            "theta": torch.zeros((n_frames, 24 + (3 if self._use_affine_mano else 0))),
            "beta": torch.zeros((n_frames, 10)),
            "rot": torch.zeros((n_frames, 6 if self._use_affine_mano else 3)),
            "trans": torch.zeros((n_frames, 3)),
        }

        if self._use_affine_mano:
            betas = (
                torch.from_numpy(
                    np.load(self._beta_paths[f"{p_num}_{grasping_hand}_betas.npy"])
                )
                .unsqueeze(0)
                .repeat(n_frames, 1)
                .float()
                .cuda()
            )
            thetas = torch.from_numpy(seq[grasping_hand]["params"]["hand_pose"]).cuda()
            rot_ax_ang = torch.from_numpy(
                seq[grasping_hand]["params"]["global_orient"]
            ).cuda()
            # rot_6d = matrix_to_rotation_6d(axis_angle_to_matrix(rot_ax_ang))
            # We'll use a dummy rotation because all the rotation we need is in the wrist, which
            # this MANO package process differently (as part of theta/hand_pose) than SMPL-X mano
            # (seprate from theta, as a global_orient vector).
            rot_6d = matrix_to_rotation_6d(
                torch.eye(3).unsqueeze(0).repeat(n_frames, 1, 1)
            ).cuda()
            trans = torch.from_numpy(seq[grasping_hand]["params"]["transl"]).cuda()
            params.update(
                {
                    "theta": torch.cat([rot_ax_ang, thetas], dim=1),
                    "beta": betas,
                    "rot": rot_6d,
                    "trans": trans,
                }
            )
        else:
            h_mesh = os.path.join(self._root_path, seq[grasping_hand]["vtemp"])
            h_vtemp = torch.from_numpy(
                np.array(o3dio.read_triangle_mesh(h_mesh).vertices)
            )  # Or with Trimesh
            smplx_params = {
                k: torch.from_numpy(v).type(torch.float32)
                for k, v in seq[grasping_hand]["params"].items()
            }
            params.update(
                {
                    "theta": smplx_params["hand_pose"],
                    "rot": smplx_params["global_orient"],
                    "trans": smplx_params["transl"],
                    "vtemp": h_vtemp,
                }
            )
        return params

    @to_cuda
    def _get_verts_and_joints(
        self,
        params: Dict[str, torch.Tensor],
        grasping_hand: str,
        pose_noise: float = 0.0,
        trans_noise: float = 0.0,
        rot_noise: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        noisy_params = {k: deepcopy(v) for k, v in params.items()}
        if self._use_affine_mano:
            noisy_params["theta"][..., :3] += rot_noise
            noisy_params["theta"][..., 3:] += pose_noise
            noisy_params["trans"] += trans_noise
            verts, joints = self._affine_mano(
                noisy_params["theta"],
                noisy_params["beta"],
                noisy_params["rot"],
                noisy_params["trans"],
            )
            faces = self._affine_mano.faces.cpu().numpy()
        else:
            with torch.no_grad(), redirect_stdout(None):
                h_m = to_cuda_(
                    smplx.create(
                        model_path=self._smplx_path,
                        model_type="mano",
                        is_rhand=grasping_hand == "rhand",
                        v_template=noisy_params["vtemp"],
                        num_pca_comps=noisy_params["theta"].shape[1],
                        flat_hand_mean=True,
                        batch_size=noisy_params["theta"].shape[0],
                    )
                )
                noisy_params["theta"] += pose_noise
                noisy_params["rot"] += rot_noise
                noisy_params["trans"] += trans_noise
                mano_result = h_m(
                    hand_pose=noisy_params["theta"],
                    global_orient=noisy_params["rot"],
                    transl=noisy_params["trans"],
                )
                faces = h_m.faces
            verts, joints = mano_result.vertices, mano_result.joints
        return verts, joints, faces, noisy_params

    def _bring_parameters_to_canonical_form(
        self, seq: Dict[str, np.ndarray], params: Dict[str, torch.Tensor]
    ):
        """
        Bring the AffineMANO or SMPL-X MANO parameters into the object frame by applying the
        inverse of the object transform to the parameters for each frame. That'll be very useful at
        test-time to initialize TTO with observations in the right coordinate frame.
        """
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
        # Apply the object transform to params so that the AffineMANO/SMPL-X parameters
        # are in the object's canonical coordinate system:
        if self._use_affine_mano:
            transform = (Transform3d().translate(params["trans"])).cuda()
            full_transform = Transform3d().compose(transform, obj_transform).cuda()
            params["trans"] = full_transform.get_matrix()[:, 3, :3]
            params["rot"] = matrix_to_rotation_6d(
                full_transform.get_matrix()[:, :3, :3]
            )
        else:
            # TODO: Fix this. I prefer to stick with AffineMANO for now since it's in line with
            # ContactPose experiments and everything is simpler. Plus, TTO on a continuous 6D
            # rotation might be easier than for axis-angle but maybe not. But if I add a MANO
            # prediction objective to the model then yeah it'd be an advantage to use rot6D.
            transform = (
                Transform3d()
                .rotate(axis_angle_to_matrix(params["rot"]))
                .translate(params["trans"])
            ).cuda()
            full_transform = Transform3d().compose(transform, obj_transform).cuda()
            params["trans"] = full_transform.get_matrix()[:, 3, :3]
            params["rot"] = matrix_to_axis_angle(full_transform.get_matrix()[:, :3, :3])
            raise NotImplementedError("SMPL-X parameters canonicalization not working")
            # (
            # gt_verts_approx,
            # gt_joints_approx,
            # _,
            # _,
            # ) = self._get_verts_and_joints(params, grasping_hand)
            # hand = Trimesh(
            # vertices=gt_verts_approx[0].cpu().numpy(),
            # faces=self._affine_mano.faces.cpu().numpy(),
            # )
            # gt_verts, _, _, _ = self._get_verts_and_joints(params, grasping_hand)
            # gt_verts = obj_transform.transform_points(gt_verts)
            # hand_gt = Trimesh(
            # vertices=gt_verts[0].cpu().numpy(),
            # faces=self._affine_mano.faces.cpu().numpy(),
            # )
            # visualize_MANO(hand, obj_mesh=obj_mesh, gt_hand=hand_gt)
        return params

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
            + f"{'affine_mano' if self._use_affine_mano else 'smplx'}_"
            + f"{'static' if self._static_grasps_only else 'dynamic'}_"
            + f"{split}",
        )
        if not osp.isdir(samples_labels_pickle_pth):
            os.makedirs(samples_labels_pickle_pth)

        choir_paths = []
        print("[*] Computing CHOIR fields...")
        dataset_mpjpe = MeanMetric()
        for mesh_pth, (grasp_name, grasp_path, frame_mask) in tqdm(
            zip(objects, grasp_sequences), total=len(objects)
        ):
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
                seq = np.load(grasp_path, allow_pickle=True)
                seq = {k: seq[k].item() for k in seq.files}
                """
                sequence:
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
                p_num = seq["sbj_id"]
                grasping_hand = "rhand"
                if not self._right_hand_only:
                    # This is how: https://github.com/kzhou23/toch/blob/019a1827e89d5fc064d30b742412524a70b92cdd/data/grab/preprocessing.py#L327
                    # Basically load sequences for each individual hands and then merge them.
                    raise NotImplementedError(
                        "Both hand grasps not implemented yet. Use right hand only for now."
                    )

                seq = self._mask_out_sequence(seq, frame_mask)

                # TODO: I still don't understand how TOCH handles approaching hands :/ It seems to
                # me that all their sequences start from the T-pose. But I may be wrong because I
                # only looked at their pre-processing code, so they might filter even more during
                # the TOCH field creation.
                # What I'm doing is just checking for the mean distance between the hand mesh and
                # the object mesh (centroids) and if it's more than a threshold (i.e. 30cm) I skip
                # the frame. In the end my grasp sequence only contains the frames where the hand
                # is within this distance to the object.

                # Let's now filter out the dynamic grasps if needed, by making sure the difference in
                # PCA space is small enough between the first and last frames. This is SUPER
                # approximative (seems to filter mostly use and pass grasps), so it'll only be used
                # for first tests on the model. Final results will be run on the full dataset with
                # dynamic grasps.
                hand_poses = torch.from_numpy(seq[grasping_hand]["params"]["hand_pose"])
                pca_dist = torch.linalg.norm(hand_poses[0] - hand_poses[-1])
                # print(f"PCA distance for {grasp_name}: {pca_dist}")
                if pca_dist > 2.0 and self._static_grasps_only:
                    continue

                obj_mesh = o3dio.read_triangle_mesh(mesh_pth)
                obj_ptcld = torch.from_numpy(
                    np.asarray(
                        obj_mesh.sample_points_uniformly(self._obj_ptcld_size).points  # type: ignore
                    )
                )
                visualize = self._debug and (random.Random().random() < 0.5)
                has_visualized = False
                # ================== Original Hand-Object Pair ==================
                gt_params = self._load_sequence_params(seq, p_num, grasping_hand)
                # Now I must center the grasp on the object s.t. the object is at the origin
                # and in canonical pose. This is done by applying the inverse of the object's
                # global orientation and translation to the hand's global orientation and
                # translation. We can use pytorch3d's transform_points for this.
                gt_params = self._bring_parameters_to_canonical_form(seq, gt_params)
                # ============ Shift the pair to the object's center ============
                if self._center_on_object_com:
                    # TODO: if self._center_on_object_com and not (self._augment and k > 0):
                    obj_center = torch.from_numpy(obj_mesh.get_center())
                    obj_mesh.translate(-obj_center)
                    obj_ptcld -= obj_center.to(obj_ptcld.device)
                    gt_params["trans"] -= obj_center.to(gt_params["trans"].device)
                # ================================================================
                gt_verts, gt_joints, gt_faces, _ = self._get_verts_and_joints(
                    gt_params, grasping_hand
                )
                n_augs = self._n_augs if self._augment else 1
                if n_augs > 1:
                    raise NotImplementedError("Augmentation not implemented yet.")
                for k in range(n_augs + 1):
                    # =================== Apply augmentation =========================
                    # TODO: Get gt_hTm first (if I manage to use AffineMANO) or implement another
                    # augmentation function for SMPLX-MANO
                    # if self._augment and k > 0:
                    # obj_mesh, gt_hTm = augment_hand_object_pose(
                    # obj_mesh, gt_hTm, around_z=False
                    # )
                    # =================================================================
                    gt_anchors = self._affine_mano.get_anchors(gt_verts)
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
                        anchor_indices=self._anchor_indices.cuda(),  # type: ignore
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
                        # Filter the sequence frames to keep only those where the hand is within
                        # 15cm of the object (as in the TOCH paper!)
                        if dist > 0.15:
                            continue

                        trans_noise = (
                            torch.randn(3, device=gt_params["trans"].device)
                            * self._perturbations[self._perturbation_level]["trans"]
                        )
                        pose_noise = torch.cat(
                            [
                                torch.randn(24, device=gt_params["theta"].device)
                                * self._perturbations[self._perturbation_level]["pca"],
                            ]
                        )
                        global_orient_noise = (
                            torch.randn(3, device=gt_params["rot"].device)
                            * self._perturbations[self._perturbation_level]["rot"]
                        )

                        gt_params_frame = {
                            k: v[i].unsqueeze(0) if k != "vtemp" else v
                            for k, v in gt_params.items()
                        }
                        verts, joints, faces, params = self._get_verts_and_joints(
                            gt_params_frame,
                            grasping_hand,
                            pose_noise=pose_noise,
                            trans_noise=trans_noise,
                            rot_noise=global_orient_noise,
                        )

                        anchors = self._affine_mano.get_anchors(verts)
                        scalar = get_scalar(anchors, obj_ptcld, self._rescale)

                        mpjpe = torch.linalg.vector_norm(
                            joints.squeeze(0) - gt_joints[i].squeeze(0), dim=1, ord=2
                        ).mean()
                        dataset_mpjpe.update(mpjpe.item())
                        # I know it's bad to do CUDA stuff in the dataset if I want to use multiple
                        # workers, but bps_torch is forcing my hand here so I might as well help it.
                        choir, rescaled_ref_pts = compute_choir(
                            to_cuda_(obj_ptcld).unsqueeze(0),
                            to_cuda_(anchors),
                            scalar=scalar,
                            bps=to_cuda_(self._bps).unsqueeze(0),  # type: ignore
                            anchor_indices=self._anchor_indices.cuda(),  # type: ignore
                            remap_bps_distances=self._remap_bps_distances,
                            exponential_map_w=self._exponential_map_w,
                        )
                        # TODO: Test pack_and_pad_sample_label work with different sizes of theta/beta! (pytorch < 2.0)
                        sample, label = pack_and_pad_sample_label(
                            params["theta"],
                            params["beta"],
                            params["rot"],
                            params["trans"],
                            choir,
                            rescaled_ref_pts,
                            scalar,
                            torch.ones((1, 1)).cuda()
                            if grasping_hand == "rhand"
                            else torch.zeros((1, 1)).cuda(),  # (1, 1)
                            gt_choir[i],
                            gt_rescaled_ref_pts,
                            gt_scalar,
                            gt_joints[i],
                            gt_anchors[i],
                            gt_params_frame["theta"],
                            gt_params_frame["beta"],
                            gt_params_frame["rot"],
                            gt_params_frame["trans"],
                            self._bps_dim,
                        )
                        # =================================================================

                        with open(sample_pth, "wb") as f:
                            pkl = pickle.dumps((sample, label, mesh_pth))
                            compressed_pkl = blosc.compress(pkl)
                            f.write(compressed_pkl)
                        choir_sequence_paths.append(sample_pth)
                        if (
                            visualize
                            and not has_visualized
                            and random.Random().random() < 0.01
                        ):
                            print(f"[*] Visualizing {grasp_name} frame {i}")
                            print("[*] Plotting CHOIR... (please be patient)")
                            # visualize_CHOIR(
                            # gt_choir[i],
                            # self._bps,
                            # gt_scalar,
                            # gt_verts[i],
                            # gt_anchors[i],
                            # obj_mesh,
                            # obj_ptcld,
                            # gt_rescaled_ref_pts.squeeze(0),
                            # self._affine_mano,
                            # )
                            gt_MANO_mesh = Trimesh(gt_verts[i].cpu().numpy(), gt_faces)
                            pred_MANO_mesh = Trimesh(
                                verts.squeeze(0).cpu().numpy(), faces
                            )
                            visualize_MANO(
                                pred_MANO_mesh, obj_mesh=obj_mesh, gt_hand=gt_MANO_mesh
                            )
                            # visualize_CHOIR_prediction(
                            # gt_choir[i].unsqueeze(0),
                            # gt_choir[i].unsqueeze(0),
                            # self._bps,
                            # self._anchor_indices,
                            # scalar,
                            # gt_scalar,
                            # rescaled_ref_pts,
                            # gt_rescaled_ref_pts,
                            # gt_verts[i].unsqueeze(0),
                            # gt_joints[i].unsqueeze(0),
                            # gt_anchors[i].unsqueeze(0),
                            # is_rhand=(grasping_hand == "rhand"),
                            # use_smplx=False,
                            # dataset="grab",
                            # remap_bps_distances=self._remap_bps_distances,
                            # exponential_map_w=self._exponential_map_w,
                            # )
                            # has_visualized = True
                    if len(choir_sequence_paths) >= 10:
                        choir_paths.append(choir_sequence_paths)
        print(
            f"[*] Dataset MPJPE (mm): {dataset_mpjpe.compute().item() * self.base_unit}"
        )
        return choir_paths

    def downsample_frames(self, seq_data, rate=2):
        # https://github.com/kzhou23/toch/blob/main/data/grab/preprocessing.py
        num_frames = int(seq_data["n_frames"])
        ones = np.ones(num_frames // rate + 1).astype(bool)
        zeros = [np.zeros(num_frames // rate + 1).astype(bool) for _ in range(rate - 1)]
        mask = np.vstack((ones, *zeros)).reshape((-1,), order="F")[:num_frames]
        return mask

    def filter_contact_frames(self, seq_data, hand: str):
        # https://github.com/kzhou23/toch/blob/main/data/grab/preprocessing.py
        """
        left/right hand not in contact
        """
        obj_contact = seq_data["contact"]["object"]

        if hand == "rhand":
            # left hand not in contact
            frame_mask = ~(
                ((obj_contact == 21) | ((obj_contact >= 26) & (obj_contact <= 40))).any(
                    axis=1
                )
            )
        else:
            # right hand not in contact
            frame_mask = ~(
                ((obj_contact == 22) | ((obj_contact >= 41) & (obj_contact <= 55))).any(
                    axis=1
                )
            )
        return frame_mask

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

        if self._use_official_splits:
            # Official splits from the TOCH paper:
            official_splits = {
                "test": [
                    "mug",
                    "wineglass",
                    "camera",
                    "binoculars",
                    "fryingpan",
                    "toothpaste",
                ],
                "val": ["apple", "toothbrush", "elephant", "hand"],
                "train": [],  # All others
            }
            if split == "train":
                object_names = [
                    o
                    for o in object_names
                    if o not in official_splits["val"] + official_splits["test"]
                ]
            elif split == "val":
                object_names = official_splits["val"]
            elif split == "test":
                object_names = official_splits["test"]
        else:
            random.Random(seed).shuffle(object_names)
            assert (self._validation_objects + self._test_objects) < len(
                object_names
            ), (
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
                    -(
                        self._validation_objects + self._test_objects
                    ) : -self._test_objects
                ]
            elif split == "test":
                object_names = object_names[-self._test_objects :]

        # 2. Load the object paths and grasp sequences paths
        print(f"[*] Loading GRAB{' (tiny)' if tiny else ''}...")
        dataset_path = osp.join(
            self._cache_dir,
            f"dataset_{split}_{n_participants}-participants"
            + (
                "_official-splits"
                if self._use_official_splits
                else f"_{self._validation_objects}-val-held-out"
                + f"_{self._test_objects}-test-held-out"
            )
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
            participants = sorted(
                [f"s{i}" for i in range(1, n_participants + 1)],
                key=lambda x: int(x[1:]),
            )
            grasping_hand = "rhand"
            if not self._right_hand_only:
                # This is how: https://github.com/kzhou23/toch/blob/019a1827e89d5fc064d30b742412524a70b92cdd/data/grab/preprocessing.py#L327
                # Basically load sequences for each individual hands and then merge them.
                raise NotImplementedError(
                    "Both hand grasps not implemented yet. Use right hand only for now."
                )
            for participant in participants:
                files = os.listdir(os.path.join(self._root_path, "grab", participant))
                for f in files:
                    if not f.endswith(".npz"):
                        continue
                    obj_name = f.split("_")[0]
                    if obj_name not in object_names:
                        continue
                    fpath = os.path.join(self._root_path, "grab", participant, f)
                    seq = np.load(fpath, allow_pickle=True)
                    seq = {k: seq[k].item() for k in seq.files}
                    hand_contact_mask = self.filter_contact_frames(seq, grasping_hand)
                    ds_mask = self.downsample_frames(seq, rate=4)  # 30FPS
                    frame_mask = ds_mask & hand_contact_mask  # Binary mask

                    obj_name = seq["obj_name"]
                    intent = seq["motion_intent"]
                    p_num = seq["sbj_id"]
                    object_path = os.path.join(
                        self._root_path, seq["object"]["object_mesh"]
                    )  # PLY files
                    assert os.path.isfile(
                        object_path
                    ), f"object_path {object_path} is not a file"

                    if self._debug:
                        scene_anim = ScenePicAnim()
                        obj_mesh = o3dio.read_triangle_mesh(object_path)
                        obj_mesh = Trimesh(
                            vertices=obj_mesh.vertices,
                            faces=obj_mesh.triangles,
                        )
                        print(
                            f"[*] Rendering base sequence {obj_name}_{p_num}_{intent}..."
                        )
                        viz_seq = self._mask_out_sequence(seq, frame_mask)
                        params = self._load_sequence_params(
                            viz_seq, p_num, grasping_hand
                        )
                        params = self._bring_parameters_to_canonical_form(
                            viz_seq, params
                        )
                        (
                            verts,
                            joints,
                            faces,
                            _,
                        ) = self._get_verts_and_joints(params, grasping_hand)
                        for i in tqdm(range(frame_mask.sum())):
                            scene_anim.add_frame(
                                {
                                    "object": obj_mesh,
                                    "hand": Trimesh(
                                        vertices=verts[i].cpu().numpy(), faces=faces
                                    ),
                                }
                            )
                        scene_anim.save_animation(
                            f"{obj_name}_{p_num}_{intent}_base_sequence.html"
                        )
                        print(
                            f"Saved base sequence as {obj_name}_{p_num}_{intent}_base_sequence.html"
                        )

                    # What happens if there are multiple grasps in the same sequence? That would mean
                    # that after masking, I get jumps in the sequence. TOCH doesn't seem to be
                    # bothering with this issue, but I think their method would struggle converging
                    # when finding these jumps. My method would definitely so let's check for gaps
                    # between True values of the mask:
                    gap_cuts = np.where(np.diff(np.where(hand_contact_mask)[0]) > 1)[0]
                    if len(gap_cuts) == 0:
                        if self._debug:
                            print(
                                f"No gaps in the mask. Saving {frame_mask.sum()} frames."
                            )
                        grasp_name = f"{obj_name}_{p_num}_{intent}"
                        grasp_sequences.append(
                            (grasp_name, fpath, frame_mask),
                        )
                        objects.append(object_path)

                    else:
                        original_length = frame_mask.shape[0]
                        if self._debug:
                            print(
                                f"Gaps in the mask of length {original_length}: ",
                                gap_cuts,
                            )
                        # Let's subdivise the sequence into multiple sequences:
                        for i, gap_end in enumerate(gap_cuts):
                            # Each gap is indicated by an index marking the end of the gap. In the
                            # case of 1 gap, we cut the sequence in 2: start -> gap and gap -> end.
                            # In the case of 2 gaps, we cut the sequence in 3: start -> gap1, gap1
                            # -> gap2, gap2 -> end. And so on.
                            if i == 0:
                                start = 0
                                end = gap_end
                            else:
                                start = gap_cuts[i - 1] + 1
                                end = gap_end
                            new_frame_mask = np.zeros_like(frame_mask).astype(bool)
                            new_frame_mask[
                                np.where(hand_contact_mask)[0][start : end + 1]
                            ] = ds_mask[np.where(hand_contact_mask)[0][start : end + 1]]
                            T = new_frame_mask.sum()
                            if T < 10:
                                continue
                            grasp_name = f"{obj_name}_{p_num}_{intent}_subseq-{i}"
                            grasp_sequences.append(
                                (grasp_name, fpath, new_frame_mask),
                            )
                            objects.append(object_path)

                            if self._debug:
                                print(
                                    f"Subsequence {i+1} of {len(gap_cuts)+1}: {T} frames (from {start} to {end})."
                                )
                                print(
                                    f"[*] Rendering subsequence {i+1} of {len(gap_cuts)+1}..."
                                )
                                scene_anim = ScenePicAnim()
                                viz_seq = self._mask_out_sequence(seq, new_frame_mask)
                                params = self._load_sequence_params(
                                    viz_seq, p_num, grasping_hand
                                )
                                params = self._bring_parameters_to_canonical_form(
                                    viz_seq, params
                                )
                                (
                                    verts,
                                    joints,
                                    faces,
                                    _,
                                ) = self._get_verts_and_joints(params, grasping_hand)
                                for j in tqdm(range(new_frame_mask.sum())):
                                    scene_anim.add_frame(
                                        {
                                            "object": obj_mesh,
                                            "hand": Trimesh(
                                                vertices=verts[j].cpu().numpy(),
                                                faces=faces,
                                            ),
                                        }
                                    )
                                scene_anim.save_animation(
                                    f"{obj_name}_{p_num}_{intent}_base_subseq-{i}.html"
                                )
                                print(
                                    f"Saved base sequence as {obj_name}_{p_num}_{intent}_base_subseq-{i}.html"
                                )

                        # Add the last subsequence:
                        start = gap_cuts[-1] + 1
                        end = len(hand_contact_mask) - 1
                        new_frame_mask = np.zeros_like(frame_mask).astype(bool)
                        new_frame_mask[
                            np.where(hand_contact_mask)[0][start:]
                        ] = ds_mask[np.where(hand_contact_mask)[0][start:]]
                        T = new_frame_mask.sum()
                        if T < 10:
                            continue
                        grasp_name = (
                            f"{obj_name}_{p_num}_{intent}_subseq-{len(gap_cuts)}"
                        )
                        grasp_sequences.append(
                            (grasp_name, fpath, new_frame_mask),
                        )
                        objects.append(object_path)

                        if self._debug:
                            print(
                                f"Subsequence {len(gap_cuts)+1} of {len(gap_cuts)+1}: {T} frames from {start} to {end}."
                            )
                            print(
                                f"[*] Rendering subsequence {len(gap_cuts)+1} of {len(gap_cuts)+1}..."
                            )
                            scene_anim = ScenePicAnim()
                            viz_seq = self._mask_out_sequence(seq, new_frame_mask)
                            params = self._load_sequence_params(
                                viz_seq, p_num, grasping_hand
                            )
                            params = self._bring_parameters_to_canonical_form(
                                viz_seq, params
                            )
                            (
                                verts,
                                joints,
                                faces,
                                _,
                            ) = self._get_verts_and_joints(params, grasping_hand)
                            for j in tqdm(range(new_frame_mask.sum())):
                                scene_anim.add_frame(
                                    {
                                        "object": obj_mesh,
                                        "hand": Trimesh(
                                            vertices=verts[j].cpu().numpy(), faces=faces
                                        ),
                                    }
                                )
                            scene_anim.save_animation(
                                f"{obj_name}_{p_num}_{intent}_base_subseq-{i+1}.html"
                            )
                            print(
                                f"Saved base sequence as {obj_name}_{p_num}_{intent}_base_subseq-{i+1}.html"
                            )
                pbar.update()
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
        dataset_name = os.path.basename(dataset_path).split(".")[0]
        return objects, grasp_sequences, dataset_name
