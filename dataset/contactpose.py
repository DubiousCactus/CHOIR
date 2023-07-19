# vim:fenc=utf-8
#
# Copyright © 2022 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
ContactPose dataset.
"""

import hashlib
import os
import os.path as osp
import pickle
import random
import sys
from contextlib import redirect_stdout
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from bps_torch.tools import sample_sphere_uniform
from hydra.utils import get_original_cwd
from open3d import io as o3dio
from pytorch3d.transforms import matrix_to_rotation_6d
from tqdm import tqdm
from trimesh import Trimesh

from conf.project import ANSI_COLORS, Theme
from dataset.base import BaseDataset
from model.affine_mano import AffineMANO
from utils import colorize, to_cuda_
from utils.dataset import compute_choir, compute_hand_object_pair_scalar
from utils.visualization import (
    visualize_CHOIR,
    visualize_CHOIR_prediction,
    visualize_MANO,
)


class ContactPoseDataset(BaseDataset):
    """ "
    A task is defined as a set of random views of the same grasp of the same object. In ContactPose
    terms, this translates to a set of random views of person X grasping object Y with the right
    hand and intent Z.
    """

    base_unit = 1000.0  # The dataset is in meters, we want to work in mm.

    def __init__(
        self,
        split: str,
        validation_objects: int = 5,
        perturbation_level: int = 0,
        obj_ptcld_size: int = 3000,
        bps_dim: int = 1024,
        n_perturbed_choir_per_sample: int = 100,
        right_hand_only: bool = True,
        center_on_object_com: bool = True,
        tiny: bool = False,
        augment: bool = False,
        seed: int = 0,
        debug: bool = False,
    ) -> None:
        self._validation_objects = validation_objects
        self._obj_ptcld_size = obj_ptcld_size
        self._cache_dir = osp.join(
            get_original_cwd(), "data", "ContactPose_preprocessed"
        )
        self._right_hand_only = right_hand_only
        self._perturbation_level = perturbation_level
        assert (
            n_perturbed_choir_per_sample > 0
        ), "n_perturbed_choir_per_sample must be > 0"
        self._n_perturbed_choir_per_sample = (
            1 if perturbation_level == 0 else n_perturbed_choir_per_sample
        )
        self._perturbations = [
            {"trans": 0.0, "rot": 0.0, "pca": 0.0},  # Level 0
            {"trans": 0.02, "rot": 0.05, "pca": 0.3},  # Level 1
            {
                "trans": 0.05,
                "rot": 0.15,
                "pca": 0.5,
            },  # Level 2 (0.05m, 0.15rad, 0.5 PCA units)
        ]

        self._bps_dim = bps_dim
        self._center_on_object_com = center_on_object_com
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
        self._bps = bps.cpu()

        super().__init__(
            dataset_root="",
            augment=augment,
            split=split,
            tiny=tiny,
            seed=seed,
            debug=debug,
        )

    @property
    def bps_dim(self) -> int:
        return self._bps_dim

    @property
    def bps(self) -> torch.Tensor:
        return self._bps

    def _load_objects_and_grasps(
        self, tiny: bool, split: str, seed: int = 0
    ) -> Tuple[dict, list, list, str]:
        if not osp.isdir(osp.join(get_original_cwd(), "vendor", "ContactPose")):
            raise RuntimeError(
                "Please clone the ContactPose dataset in the vendor/ directory as 'ContactPose'"
            )
        sys.path.append(osp.join(get_original_cwd(), "vendor", "ContactPose"))
        sys.path.append(
            osp.join(get_original_cwd(), "vendor", "ContactPose", "thirdparty", "mano")
        )
        sys.path.append(
            osp.join(
                get_original_cwd(),
                "vendor",
                "ContactPose",
                "thirdparty",
                "mano",
                "webuser",
            )
        )
        from vendor.ContactPose.utilities.dataset import ContactPose, get_object_names

        def load_contact_pose_data(
            p_num: int, intent: str, object_name: str, hand_idx: int
        ) -> Optional[Tuple]:
            with redirect_stdout(None):
                cp = ContactPose(p_num, intent, object_name)
            random_grasp_frame = (
                cp.mano_params[hand_idx],
                cp.mano_meshes()[hand_idx],
            )  # {'pose': _, 'betas': _, 'hTm': _} for mano, {'vertices': _, 'faces': _, 'joints': _} for mesh
            if random_grasp_frame[0] is None or random_grasp_frame[1] is None:
                # Some participants don't manipulate all objects :/
                # print(f"[!] Couldn't load {p_num}, {intent}, {object_name}")
                return None
            return (
                cp.contactmap_filename,
                random_grasp_frame,
            )

        # First, build a dictionary of object names to the participant, intent, and hand used. The
        # reason we don't do it all in one pass is that some participants may not manipulate some
        # objects.
        cp_dataset = {}
        n_participants = 15 if tiny else 51
        p_nums = list(range(1, n_participants))
        intents = ["use", "handoff"]
        if not osp.isdir(self._cache_dir):
            os.makedirs(self._cache_dir)
        for p_num in p_nums:
            for intent in intents:
                for obj_name in get_object_names(p_num, intent):
                    if obj_name not in cp_dataset:
                        cp_dataset[obj_name] = []
                    cp_dataset[obj_name].append((p_num, intent, "right"))
                    if not self._right_hand_only:
                        cp_dataset[obj_name].append((p_num, intent, "left"))
        hand_indices = {"right": 1, "left": 0}

        # Keep only the first n_objects - validation_objects if we're in training mode, and the last
        # validation_objects if we're in eval mode:
        object_names = sorted(list(cp_dataset.keys()))
        random.Random(seed).shuffle(object_names)
        assert self._validation_objects < len(object_names), (
            f"validation_objects ({self._validation_objects})"
            + " must be less than n_objects ({len(object_names)})"
        )
        assert (
            self._validation_objects >= 1
        ), f"validation_objects ({self._validation_objects}) must be greater or equal to 1"
        if split == "train":
            object_names = object_names[: -self._validation_objects]
        else:
            object_names = object_names[-self._validation_objects :]
        # Now, build the dataset.
        # objects: unique object paths
        # objects_w_contacts: object with contacts paths
        # grasps: MANO parameters w/ global pose pickle paths
        objects, objects_w_contacts, grasps = {}, [], []
        print(f"[*] Loading ContactPose{' (tiny)' if tiny else ''}...")

        dataset_path = osp.join(
            self._cache_dir,
            f"dataset_{split}_{n_participants}-participants"
            + f"_{self._validation_objects}-hold-out"
            + f"_{self._obj_ptcld_size}-obj-pts"
            + f"{'right-hand' if self._right_hand_only else 'both-hands'}_seed-{seed}.pkl",
        )
        if osp.isfile(dataset_path):
            objects, objects_w_contacts, grasps = pickle.load(open(dataset_path, "rb"))
        else:
            for obj_name in tqdm(object_names):
                for p_num, intent, hand in cp_dataset[obj_name]:
                    # Generate the data since the dataset file does not exist.
                    grasp_path = osp.join(
                        self._cache_dir,
                        f"{obj_name}_{p_num}_{intent}_{hand}-hand.pkl",
                    )
                    data = load_contact_pose_data(
                        p_num, intent, obj_name, hand_indices[hand]
                    )
                    if data is None:
                        continue
                    obj_mesh_w_contacts_path, grasp = data
                    with open(grasp_path, "wb") as f:
                        # Log participant no and intent in the data so we can debug later on.
                        pickle.dump(
                            {
                                "grasp": grasp,
                                "p_num": p_num,
                                "intent": intent,
                                "obj_name": obj_name,
                                "hand_idx": hand,
                            },
                            f,
                        )
                    objects_w_contacts.append(obj_mesh_w_contacts_path)
                    grasps.append(grasp_path)
                    if obj_name not in objects:
                        objects[obj_name] = obj_mesh_w_contacts_path
            pickle.dump((objects, objects_w_contacts, grasps), open(dataset_path, "wb"))
        print(
            f"[*] Loaded {len(object_names)} objects and {len(grasps)} grasp sequences"
        )
        print(
            colorize(
                f"[*] {'Training' if split == 'train' else 'Validation'} objects: {', '.join(object_names)}",
                ANSI_COLORS[
                    Theme.TRAINING.value if split == "train" else Theme.VALIDATION.value
                ],
            )
        )
        assert len(objects_w_contacts) == len(grasps)
        return (
            objects,
            objects_w_contacts,
            grasps,
            osp.basename(dataset_path.split(".")[0]),
        )

    def _load(
        self,
        dataset_root: str,
        tiny: bool,
        split: str,
        objects: Dict[str, str],
        objects_w_contacts: List[str],
        grasps: List,
        dataset_name: str,
    ) -> List[str]:
        sample_paths = []
        """
        Make sure that we're using the same BPS for CHOIR generation, training and test-time
        optimization! It's very important otherwise we can't learn anything meaningful and
        generalize.
        """
        # TODO: We should just hash all the class properties and use that as the cache key. This is
        # a bit hacky and not scalable.
        samples_labels_pickle_pth = osp.join(
            self._cache_dir,
            "samples_and_labels",
            f"dataset_{hashlib.shake_256(dataset_name.encode()).hexdigest(8)}_"
            + f"perturbed-{self._perturbation_level}_"
            + f"{self._bps_dim}-bps_"
            + f"{'object-centered_' if self._center_on_object_com else ''}"
            + f"{split}",
        )
        if not osp.isdir(samples_labels_pickle_pth):
            os.makedirs(samples_labels_pickle_pth)
        affine_mano: AffineMANO = to_cuda_(AffineMANO())  # type: ignore

        # For each object-grasp pair, compute the CHOIR field.
        print("[*] Computing CHOIR fields...")
        for mesh_pth, grasp_pth in tqdm(
            zip(objects_w_contacts, grasps), total=len(objects_w_contacts)
        ):
            # Load the object mesh and MANO params
            with open(grasp_pth, "rb") as f:
                grasp_data = pickle.load(f)
            """
            {
                'grasp': (
                    {'pose': _, 'betas': _, 'hTm': _},
                    {'vertices': _, 'faces': _, 'joints': _}
                ),
                'p_num': _, 'intent': _, 'obj_name': _, 'hand_idx': _
            }
            """
            obj_name, p_num, intent, hand_idx = (
                grasp_data["obj_name"],
                grasp_data["p_num"],
                grasp_data["intent"],
                grasp_data["hand_idx"],
            )
            grasp_name = f"{obj_name}_{p_num}_{intent}_{hand_idx}"
            if not osp.isdir(osp.join(samples_labels_pickle_pth, grasp_name)):
                os.makedirs(osp.join(samples_labels_pickle_pth, grasp_name))
            if (
                len(os.listdir(osp.join(samples_labels_pickle_pth, grasp_name)))
                >= self._n_perturbed_choir_per_sample
            ):
                sample_paths += [
                    osp.join(
                        samples_labels_pickle_pth, grasp_name, f"sample_{i:06d}.pkl"
                    )
                    for i in range(self._n_perturbed_choir_per_sample)
                ]
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
                mano_params = grasp_data["grasp"][0]
                gt_hTm = (
                    torch.from_numpy(mano_params["hTm"]).float().unsqueeze(0).cuda()
                )
                gt_rot_6d = matrix_to_rotation_6d(gt_hTm[:, :3, :3])
                gt_trans = gt_hTm[:, :3, 3]
                gt_theta, gt_beta = (
                    torch.tensor(mano_params["pose"]).unsqueeze(0).cuda(),
                    torch.tensor(mano_params["betas"]).unsqueeze(0).cuda(),
                )
                # ============ Shift the pair to the object's center ============
                if self._center_on_object_com:
                    obj_center = torch.from_numpy(obj_mesh.get_center())
                    obj_mesh.translate(-obj_center)
                    obj_ptcld -= obj_center.to(obj_ptcld.device)
                    gt_trans -= obj_center.to(gt_trans.device)
                # ===============================================================
                gt_verts, gt_joints = affine_mano(
                    gt_theta, gt_beta, gt_rot_6d, gt_trans
                )
                # ===============================================================
                gt_anchors = affine_mano.get_anchors(gt_verts)
                # ================== Rescaled Hand-Object Pair ==================
                gt_scalar = compute_hand_object_pair_scalar(gt_anchors, obj_ptcld)

                # I know it's bad to do CUDA stuff in the dataset if I want to use multiple
                # workers, but bps_torch is forcing my hand here so I might as well help it.
                gt_choir, gt_rescaled_ref_pts = compute_choir(
                    to_cuda_(obj_ptcld).unsqueeze(0),
                    to_cuda_(gt_anchors),
                    scalar=gt_scalar,
                    bps=to_cuda_(self._bps).unsqueeze(0),  # type: ignore
                )

                for j in range(self._n_perturbed_choir_per_sample):
                    sample_pth = osp.join(
                        samples_labels_pickle_pth, grasp_name, f"sample_{j:06d}.pkl"
                    )
                    if osp.isfile(sample_pth):
                        sample_paths.append(sample_pth)
                        continue

                    theta, beta, rot_6d, trans = (
                        deepcopy(gt_theta),
                        deepcopy(gt_beta),
                        deepcopy(gt_rot_6d),
                        deepcopy(gt_trans),
                    )
                    if self._perturbation_level > 0:
                        trans_noise = (
                            torch.rand(3, device=trans.device)
                            * self._perturbations[self._perturbation_level]["trans"]
                        )
                        pose_noise = torch.cat(
                            [
                                torch.rand(3)
                                * self._perturbations[self._perturbation_level]["rot"],
                                torch.rand(15)
                                * self._perturbations[self._perturbation_level]["pca"],
                            ]
                        ).to(theta.device)
                        theta += pose_noise
                        trans += trans_noise

                    verts, _ = affine_mano(theta, beta, rot_6d, trans)

                    anchors = affine_mano.get_anchors(verts)
                    scalar = compute_hand_object_pair_scalar(anchors, obj_ptcld)
                    # I know it's bad to do CUDA stuff in the dataset if I want to use multiple
                    # workers, but bps_torch is forcing my hand here so I might as well help it.
                    choir, rescaled_ref_pts = compute_choir(
                        to_cuda_(obj_ptcld).unsqueeze(0),
                        to_cuda_(anchors),
                        scalar=scalar,
                        bps=to_cuda_(self._bps).unsqueeze(0),  # type: ignore
                    )
                    # anchor_orientations = torch.nn.functional.normalize(
                    # anchor_deltas, dim=1
                    # )
                    # Compute the dense MANO contact map
                    # hand_contacts = compute_hand_contacts_simple(
                    # ref_pts.float(), verts.float()
                    # )

                    sample = (
                        choir.squeeze(0).cpu(),
                        rescaled_ref_pts.squeeze(0).cpu(),
                        scalar.cpu(),
                    )
                    label = (
                        gt_choir.squeeze(0).cpu(),
                        # anchor_orientations,
                        gt_rescaled_ref_pts.squeeze(0).cpu(),
                        gt_scalar.cpu(),
                        gt_joints.squeeze(0).cpu(),
                        gt_anchors.squeeze(0).cpu(),
                        gt_theta.squeeze(0).cpu(),
                        gt_beta.squeeze(0).cpu(),
                        gt_rot_6d.squeeze(0).cpu(),
                        gt_trans.squeeze(0).cpu(),
                    )

                    for v in sample:
                        if isinstance(v, torch.Tensor):
                            assert v.device == torch.device("cpu")
                    for v in label:
                        if isinstance(v, torch.Tensor):
                            assert v.device == torch.device("cpu")
                    with open(sample_pth, "wb") as f:
                        pickle.dump((sample, label), f)
                    sample_paths.append(sample_pth)
                    if visualize and not has_visualized:
                        print("[*] Plotting CHOIR... (please be patient)")
                        # hand_contacts = (
                        # compute_hand_contacts_simple(ref_pts.float(), verts.float())
                        # .squeeze(0)
                        # .cpu()
                        # )
                        visualize_CHOIR(
                            gt_choir.squeeze(0),
                            self._bps,
                            gt_scalar,
                            # hand_contacts,
                            gt_verts.squeeze(0),
                            gt_anchors.squeeze(0),
                            # anchor_orientations,
                            obj_mesh,
                            obj_ptcld,
                            rescaled_ref_pts.squeeze(0),
                            affine_mano,
                        )
                        faces = affine_mano.faces
                        gt_MANO_mesh = Trimesh(
                            gt_verts.squeeze(0).cpu().numpy(), faces.cpu().numpy()
                        )
                        visualize_MANO(
                            verts, faces, obj_mesh=obj_mesh, gt_hand=gt_MANO_mesh
                        )
                        visualize_CHOIR_prediction(
                            choir,
                            gt_choir,
                            self._bps,
                            scalar.unsqueeze(0),
                            gt_scalar.unsqueeze(0),
                            rescaled_ref_pts,
                            gt_rescaled_ref_pts,
                            {
                                "pose": gt_theta,
                                "beta": gt_beta,
                                "rot_6d": gt_rot_6d,
                                "trans": gt_trans,
                            },
                            self._bps_dim,
                        )
                        has_visualized = True
        return sample_paths
