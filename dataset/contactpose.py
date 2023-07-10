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
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from hydra.utils import get_original_cwd
from manotorch.anchorlayer import AnchorLayer
from open3d import io as o3dio
from pytorch3d.transforms import matrix_to_rotation_6d
from tqdm import tqdm

from conf.project import ANSI_COLORS, Theme
from dataset.base import BaseDataset
from model.affine_mano import AffineMANO
from utils import colorize, to_cuda_
from utils.dataset import compute_choir, compute_hand_contacts_simple
from utils.visualization import visualize_CHOIR


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
        perturbation_level: float = 0,
        obj_ptcld_size: int = 3000,
        bps_dim: int = 1024,
        anchor_assignment: str = "random",
        scaling: str = "none",
        unit_cube: bool = True,
        positive_unit_cube: bool = False,
        right_hand_only: bool = True,
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
        if positive_unit_cube:
            assert unit_cube, "positive_unit_cube=True requires unit_cube=True"
        self._positive_unit_cube = positive_unit_cube
        self._unit_cube = unit_cube
        self._perturbation_level = perturbation_level
        self._bps_dim = bps_dim
        self._anchor_assignment = anchor_assignment
        super().__init__(
            dataset_root="",
            augment=augment,
            split=split,
            scaling=scaling,
            tiny=tiny,
            seed=seed,
            debug=debug,
        )

    @property
    def bps_dim(self) -> int:
        return self._bps_dim

    @property
    def anchor_assignment(self) -> str:
        return self._anchor_assignment

    def _load_objects_and_grasps(
        self, tiny: bool, split: str, seed: int = 0
    ) -> Tuple[list, list, str]:
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
        n_participants = 35 if tiny else 51
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
        objects = []
        grasps = []
        print(f"[*] Loading ContactPose{' (tiny)' if tiny else ''}...")

        dataset_path = osp.join(
            self._cache_dir,
            f"dataset_{split}_{n_participants}-participants"
            + f"_{self._validation_objects}-hold-out"
            + f"_{self._obj_ptcld_size}-obj-pts"
            + f"{'right-hand' if self._right_hand_only else 'both-hands'}_seed-{seed}.pkl",
        )
        if osp.isfile(dataset_path):
            objects, grasps = pickle.load(open(dataset_path, "rb"))
        else:
            if not osp.isfile(osp.join(self._cache_dir, "missing.txt")):
                with open(osp.join(self._cache_dir, "missing.txt"), "w") as f:
                    f.write("")
            for obj_name in tqdm(object_names):
                for p_num, intent, hand in cp_dataset[obj_name]:
                    # Generate the data since the dataset file does not exist.
                    grasp_path = osp.join(
                        self._cache_dir,
                        f"{obj_name}_{p_num}_{intent}_{hand}-hand.pkl",
                    )
                    # Make sure the hand_pose_seq_pth is not in the list of failures in missing.txt.
                    # I know it's a very stupid way of doing it but because I load the dataset
                    # twice (train and eval) I don't want to have to recompute the missing ones, and
                    # I can't store them in memory because I instantiate this class twice.
                    missing = False
                    with open(osp.join(self._cache_dir, "missing.txt"), "r") as f:
                        for line in f:
                            if grasp_path in line:
                                missing = True
                                break
                    if missing:
                        continue
                    data = load_contact_pose_data(
                        p_num, intent, obj_name, hand_indices[hand]
                    )
                    if data is None:
                        with open(osp.join(self._cache_dir, "missing.txt"), "a") as f:
                            f.write(grasp_path + "\n")
                        continue
                    obj_mesh_path, grasp = data
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
                    objects.append(obj_mesh_path)
                    grasps.append(grasp_path)
            pickle.dump((objects, grasps), open(dataset_path, "wb"))
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
        assert len(objects) == len(grasps)
        return objects, grasps, osp.basename(dataset_path.split(".")[0])

    def _load(
        self,
        dataset_root: str,
        tiny: bool,
        split: str,
        objects: List[str],
        grasps: List,
        dataset_name: str,
    ) -> Tuple[Union[dict, list], Union[dict, list]]:
        samples, labels = [], []
        samples_labels_pickle_pth = osp.join(
            self._cache_dir,
            "samples_and_labels",
            f"dataset_{hashlib.shake_256(dataset_name.encode()).hexdigest(8)}_"
            + f"perturbed-{self._perturbation_level}_"
            + f"{self._anchor_assignment}-assigned_"
            + f"{self._bps_dim}-bps_"
            + f"{split}.pkl",
        )
        if not osp.isdir(osp.dirname(samples_labels_pickle_pth)):
            os.makedirs(osp.dirname(samples_labels_pickle_pth))
        if osp.isfile(samples_labels_pickle_pth):
            with open(samples_labels_pickle_pth, "rb") as f:
                samples, labels = pickle.load(f)
        else:
            affine_mano: AffineMANO = to_cuda_(AffineMANO())  # type: ignore
            anchor_layer = AnchorLayer(
                anchor_root="vendor/manotorch/assets/anchor"
            ).cuda()
            # First of all, compute the pointclouds mean for all the objects in the dataset.
            pointclouds = []
            meshes = []
            print("[*] Computing object pointclouds mean...")
            for object_with_contacts_pth in objects:
                obj_mesh = o3dio.read_triangle_mesh(object_with_contacts_pth)
                obj_ptcld = torch.from_numpy(
                    np.asarray(
                        obj_mesh.sample_points_uniformly(self._obj_ptcld_size).points  # type: ignore
                    )
                )
                pointclouds.append(obj_ptcld)
                meshes.append(obj_mesh)  # For visualization
            pointclouds_mean = torch.stack(pointclouds, dim=0).mean(dim=0).mean(dim=0)
            # For each object-grasp pair, compute the CHOIR field.
            # TODO: Vectorize this after loading grasps one by one. Or not? It's more convenient to
            # have lists of samples than a single big tensor. Eeeh I don't need efficiency here.
            has_visualized = False
            print("[*] Computing CHOIR fields...")
            for object_ptcld, mesh, grasp_pth in zip(pointclouds, meshes, grasps):
                # Load the object mesh and MANO params
                grasp_data = pickle.load(open(grasp_pth, "rb"))
                """
                {
                    'grasp': (
                        {'pose': _, 'betas': _, 'hTm': _},
                        {'vertices': _, 'faces': _, 'joints': _}
                    ),
                    'p_num': _, 'intent': _, 'obj_name': _, 'hand_idx': _
                }
                """
                mano_params = grasp_data["grasp"][0]
                hTm = torch.from_numpy(mano_params["hTm"]).float().unsqueeze(0).cuda()
                rot_6d = matrix_to_rotation_6d(hTm[:, :3, :3])
                trans = hTm[:, :3, 3]
                verts, joints = affine_mano(
                    torch.tensor(mano_params["pose"]).unsqueeze(0).cuda(),
                    torch.tensor(mano_params["betas"]).unsqueeze(0).cuda(),
                    rot_6d,
                    trans,
                )

                # Rescale the meshes to fit in a unit cube if scaling is enabled
                if self._scaling != "none":
                    raise NotImplementedError

                # Compute the CHOIR field
                anchors = anchor_layer(verts)
                choir, pcl_mean, pcl_scalar, ref_pts = compute_choir(
                    to_cuda_(object_ptcld),
                    to_cuda_(anchors),
                    pointclouds_mean=to_cuda_(pointclouds_mean),
                    bps_dim=self._bps_dim,  # type: ignore
                )
                # Compute the dense MANO contact map
                hand_contacts = compute_hand_contacts_simple(
                    ref_pts.float(), verts.float()
                )

                # Remove batch dimension
                choir = choir.squeeze(0)
                pcl_mean = pcl_mean.squeeze(0)
                pcl_scalar = pcl_scalar.squeeze(0)
                hand_contacts = hand_contacts.squeeze(0)
                joints = joints.squeeze(0)
                anchors = anchors.squeeze(0)
                rot_6d = rot_6d.squeeze(0)
                trans = trans.squeeze(0)

                samples.append(
                    {
                        "noisy_choir": choir,
                        "noisy_hand_contacts": hand_contacts,
                        "pcl_mean": pcl_mean,
                        "pcl_scalar": pcl_scalar,
                        "bps_dim": self._bps_dim,
                    }
                )
                labels.append(
                    {
                        "choir": choir,
                        "hand_contacts": hand_contacts,
                        "joints": joints,
                        "anchors": anchors,
                        "mano_params": {
                            "pose": torch.tensor(mano_params["pose"]),
                            "betas": torch.tensor(mano_params["betas"]),
                            "rot_6d": rot_6d,
                            "trans": trans,
                        },
                    }
                )
                if self._debug and not has_visualized and (random.random() < 0.25):
                    visualize_CHOIR(
                        choir,
                        hand_contacts,
                        verts,
                        anchors,
                        mesh,
                        object_ptcld,
                        ref_pts,
                        affine_mano,
                    )
                    has_visualized = True
            with open(samples_labels_pickle_pth, "wb") as f:
                pickle.dump((samples, labels), f)
        return samples, labels
