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
from typing import List, Optional, Tuple

import numpy as np
import torch
from hydra.utils import get_original_cwd
from open3d import io as o3dio
from pytorch3d.transforms import matrix_to_rotation_6d, random_rotation
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
    terms, this translates to a set of random views of person X grasping object Y with hand H and
    intent Z.
    """

    base_unit = 1000.0  # The dataset is in meters, we want to work in mm.

    def __init__(
        self,
        split: str,
        validation_objects: int = 3,
        test_objects: int = 2,
        perturbation_level: int = 0,
        obj_ptcld_size: int = 3000,
        bps_dim: int = 1024,
        noisy_samples_per_grasp: int = 100,
        max_views_per_grasp: int = 5,
        right_hand_only: bool = True,
        center_on_object_com: bool = True,
        tiny: bool = False,
        augment: bool = False,
        n_augs: int = 10,
        seed: int = 0,
        debug: bool = False,
        rescale: str = "none",
        remap_bps_distances: bool = False,
        exponential_map_w: float = 5.0,
    ) -> None:
        assert max_views_per_grasp <= noisy_samples_per_grasp
        assert max_views_per_grasp > 0
        assert noisy_samples_per_grasp > 0, "noisy_samples_per_grasp must be > 0"
        self._perturbations = [
            {"trans": 0.0, "rot": 0.0, "pca": 0.0},  # Level 0
            {"trans": 0.02, "rot": 0.05, "pca": 0.3},  # Level 1
            {
                "trans": 0.05,
                "rot": 0.15,
                "pca": 0.5,
            },  # Level 2 (0.05m, 0.15rad, 0.5 PCA units)
        ]
        super().__init__(
            dataset_name="ContactPose",
            bps_dim=bps_dim,
            validation_objects=validation_objects,
            test_objects=test_objects,
            right_hand_only=right_hand_only,
            obj_ptcld_size=obj_ptcld_size,
            perturbation_level=perturbation_level,
            max_views_per_grasp=max_views_per_grasp,
            noisy_samples_per_grasp=noisy_samples_per_grasp,
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
        # Now, build the dataset.
        # objects: unique object paths
        # objects_w_contacts: object with contacts paths
        # grasps: MANO parameters w/ global pose pickle paths
        objects_w_contacts, grasps = [], []
        print(f"[*] Loading ContactPose{' (tiny)' if tiny else ''}...")

        dataset_path = osp.join(
            self._cache_dir,
            f"dataset_{split}_{n_participants}-participants"
            + f"_{self._validation_objects}-val-held-out"
            + f"_{self._test_objects}-test-held-out"
            + f"_{self._obj_ptcld_size}-obj-pts"
            + f"{'right-hand' if self._right_hand_only else 'both-hands'}_seed-{seed}.pkl",
        )
        if osp.isfile(dataset_path):
            objects_w_contacts, grasps = pickle.load(open(dataset_path, "rb"))
        else:
            # TODO: Refactor this? I think we don't need all that code and dumping into independent
            # files! It's probably light enough in memory that we can dump it all in one file. Also
            # Python being Python, this loop is horribly slow.
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
            pickle.dump((objects_w_contacts, grasps), open(dataset_path, "wb"))
        n_left = sum([1 for g in grasps if "left-hand" in g])
        n_right = sum([1 for g in grasps if "right-hand" in g])

        print(
            f"[*] Loaded {len(object_names)} objects and {len(grasps)} grasp sequences ({n_left} left hand, {n_right} right hand)"
        )
        split_name = (
            "train"
            if split == "train"
            else ("validation" if split == "val" else "Test")
        )
        print(
            colorize(
                f"[*] {split_name} objects: {', '.join(object_names)}",
                ANSI_COLORS[
                    Theme.TRAINING.value if split == "train" else Theme.VALIDATION.value
                ],
            )
        )
        assert len(objects_w_contacts) == len(grasps)
        return (
            objects_w_contacts,
            grasps,
            osp.basename(dataset_path.split(".")[0]),
        )

    def _load(
        self,
        split: str,
        objects_w_contacts: List[str],
        grasps: List,
        dataset_name: str,
    ) -> List[str]:
        grasp_paths = []
        # TODO: We should just hash all the class properties and use that as the cache key. This is
        # a bit hacky and not scalable.
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
            + f"{split}{'_augmented' if self._augment else ''}",
        )
        if not osp.isdir(samples_labels_pickle_pth):
            os.makedirs(samples_labels_pickle_pth)
        affine_mano: AffineMANO = to_cuda_(AffineMANO())  # type: ignore

        n_augs = self._n_augs if self._augment else 1

        # For each object-grasp pair, compute the CHOIR field.
        print("[*] Computing CHOIR fields...")
        pbar = tqdm(total=len(objects_w_contacts) * n_augs)
        for mesh_pth, grasp_pth in zip(objects_w_contacts, grasps):
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
            for k in range(n_augs):
                grasp_name = f"{obj_name}_{p_num}_{intent}_{hand_idx}_aug-{k}"
                if not osp.isdir(osp.join(samples_labels_pickle_pth, grasp_name)):
                    os.makedirs(osp.join(samples_labels_pickle_pth, grasp_name))
                if (
                    len(os.listdir(osp.join(samples_labels_pickle_pth, grasp_name)))
                    >= self._noisy_samples_per_grasp
                ):
                    grasp_paths.append(
                        [
                            osp.join(
                                samples_labels_pickle_pth,
                                grasp_name,
                                f"sample_{i:06d}.pkl",
                            )
                            for i in range(self._noisy_samples_per_grasp)
                        ]
                    )
                else:
                    visualize = self._debug and (random.Random().random() < 0.05)
                    has_visualized = False
                    # ================== Original Hand-Object Pair ==================
                    mano_params = grasp_data["grasp"][0]
                    gt_hTm = (
                        torch.from_numpy(mano_params["hTm"]).float().unsqueeze(0).cuda()
                    )
                    obj_mesh = o3dio.read_triangle_mesh(mesh_pth)
                    # =================== Apply augmentation =========================
                    if self._augment:
                        # Randomly rotate the object and hand meshes
                        R = random_rotation().to(gt_hTm.device)
                        # It is CRUCIAL to translate both to the center of the object before rotating, because the hand joints
                        # are expressed w.r.t. the object center. Otherwise, weird things happen.
                        rotate_origin = obj_mesh.get_center()
                        obj_mesh.translate(-rotate_origin)
                        # Rotate the object and hand
                        obj_mesh.rotate(R.cpu().numpy(), np.array([0, 0, 0]))
                        r_hTm = torch.eye(4)
                        # We need to rotate the 4x4 MANO root pose as well, by first converting R to a
                        # 4x4 homogeneous matrix so we can apply it to the 4x4 pose matrix:
                        R4 = torch.eye(4)
                        R4[:3, :3] = R.float()
                        gt_hTm[:, :3, 3] -= (
                            torch.from_numpy(rotate_origin).to(gt_hTm.device).float()
                        )
                        r_hTm = R4.to(gt_hTm.device) @ gt_hTm
                        gt_hTm = r_hTm

                    # =================================================================
                    gt_rot_6d = matrix_to_rotation_6d(gt_hTm[:, :3, :3])
                    gt_trans = gt_hTm[:, :3, 3]
                    gt_theta, gt_beta = (
                        torch.tensor(mano_params["pose"]).unsqueeze(0).cuda(),
                        torch.tensor(mano_params["betas"]).unsqueeze(0).cuda(),
                    )
                    obj_ptcld = (
                        torch.from_numpy(
                            np.asarray(
                                obj_mesh.sample_points_uniformly(self._obj_ptcld_size).points  # type: ignore
                            )
                        )
                        .cuda()
                        .float()
                    )
                    # ============ Shift the pair to the object's center ============
                    # When we augment we necessarily recenter on the object, so we don't need to do it here.
                    if self._center_on_object_com and not self._augment:
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
                    if self._rescale == "pair":
                        gt_scalar = compute_hand_object_pair_scalar(
                            gt_anchors, obj_ptcld
                        )
                    elif self._rescale == "object":
                        gt_scalar = compute_hand_object_pair_scalar(
                            gt_anchors, obj_ptcld, scale_by_object_only=True
                        )
                    elif self._rescale == "fixed":
                        gt_scalar = torch.tensor([10.0]).cuda()
                    elif self._rescale == "none":
                        gt_scalar = torch.tensor([1.0]).cuda()
                    else:
                        raise ValueError(f"Unknown rescale type {self._rescale}")

                    # I know it's bad to do CUDA stuff in the dataset if I want to use multiple
                    # workers, but bps_torch is forcing my hand here so I might as well help it.
                    gt_choir, gt_rescaled_ref_pts = compute_choir(
                        to_cuda_(obj_ptcld).unsqueeze(0),
                        to_cuda_(gt_anchors),
                        scalar=gt_scalar,
                        bps=to_cuda_(self._bps).unsqueeze(0),  # type: ignore
                        remap_bps_distances=self._remap_bps_distances,
                        exponential_map_w=self._exponential_map_w,
                    )

                    sample_paths = []
                    # ================== Perturbed Hand-Object Pair ==================
                    for j in range(self._noisy_samples_per_grasp):
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
                                    * self._perturbations[self._perturbation_level][
                                        "rot"
                                    ],
                                    torch.rand(15)
                                    * self._perturbations[self._perturbation_level][
                                        "pca"
                                    ],
                                ]
                            ).to(theta.device)
                            theta += pose_noise
                            trans += trans_noise

                        verts, _ = affine_mano(theta, beta, rot_6d, trans)

                        anchors = affine_mano.get_anchors(verts)
                        if self._rescale == "pair":
                            scalar = compute_hand_object_pair_scalar(anchors, obj_ptcld)
                        elif self._rescale == "object":
                            scalar = compute_hand_object_pair_scalar(
                                anchors, obj_ptcld, scale_by_object_only=True
                            )
                        elif self._rescale == "fixed":
                            scalar = torch.tensor([10.0]).cuda()
                        elif self._rescale == "none":
                            scalar = torch.tensor([1.0]).cuda()
                        else:
                            raise ValueError(f"Unknown rescale type {self._rescale}")
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

                        if "2.0" in torch.__version__:
                            # ============ With Pytorch 2.0 Nested Tensor ============
                            sample = torch.nested.to_padded_tensor(
                                torch.nested.nested_tensor(
                                    [
                                        choir.squeeze(0),  # (N, 2)
                                        rescaled_ref_pts.squeeze(0),  # (N, 3)
                                        scalar.unsqueeze(0),  # (1, 1)
                                        torch.ones((1, 1)).cuda()
                                        if hand_idx == "right"
                                        else torch.zeros((1, 1)).cuda(),  # (1, 1)
                                    ]
                                ),
                                0.0,
                            ).cpu()

                            label = torch.nested.to_padded_tensor(
                                torch.nested.nested_tensor(
                                    [
                                        gt_choir.squeeze(0),  # (N, 2)
                                        gt_rescaled_ref_pts.squeeze(0),  # (N, 3)
                                        gt_scalar.unsqueeze(0),  # (1, 1)
                                        gt_joints.squeeze(0),  # (21, 3)
                                        gt_anchors.squeeze(0),  # (32, 3)
                                        gt_theta,  # (1, 18)
                                        gt_beta,  # (1, 10)
                                        gt_rot_6d,  # (1, 6)
                                        gt_trans,  # (1, 3)
                                    ]
                                ),
                                0.0,
                            ).cpu()
                            # ========================================================
                        else:
                            # ============== Without Pytorch 2.0 Nested Tensor ==============
                            sample = torch.stack(
                                [
                                    torch.nn.functional.pad(
                                        choir.squeeze(0), (0, 1), value=0.0
                                    ),
                                    rescaled_ref_pts.squeeze(0),
                                    torch.nn.functional.pad(
                                        scalar.unsqueeze(0), (0, 2), value=0.0
                                    ).repeat((self._bps_dim, 1)),
                                    torch.nn.functional.pad(
                                        (
                                            torch.ones((1, 1)).cuda()
                                            if hand_idx == "right"
                                            else torch.zeros((1, 1)).cuda()
                                        ),
                                        (0, 2),
                                        value=0.0,
                                    ).repeat((self._bps_dim, 1)),
                                ],
                                dim=0,
                            ).cpu()

                            padded_joints = torch.zeros(
                                (self._bps_dim, 18), device=gt_joints.device
                            )
                            padded_joints[
                                : gt_joints.squeeze(0).shape[0], :3
                            ] = gt_joints.squeeze(0)
                            padded_anchors = torch.zeros(
                                (self._bps_dim, 18), device=gt_anchors.device
                            )
                            padded_anchors[
                                : gt_anchors.squeeze(0).shape[0], :3
                            ] = gt_anchors.squeeze(0)
                            label = torch.stack(
                                [
                                    torch.nn.functional.pad(
                                        gt_choir.squeeze(0), (0, 16), value=0.0
                                    ),
                                    torch.nn.functional.pad(
                                        gt_rescaled_ref_pts.squeeze(0),
                                        (0, 15),
                                        value=0.0,
                                    ),
                                    torch.nn.functional.pad(
                                        gt_scalar.unsqueeze(0), (0, 17), value=0.0
                                    ).repeat((self._bps_dim, 1)),
                                    padded_joints,
                                    padded_anchors,
                                    gt_theta.repeat((self._bps_dim, 1)),
                                    torch.nn.functional.pad(
                                        gt_beta, (0, 8), value=0.0
                                    ).repeat((self._bps_dim, 1)),
                                    torch.nn.functional.pad(
                                        gt_rot_6d, (0, 12), value=0.0
                                    ).repeat((self._bps_dim, 1)),
                                    torch.nn.functional.pad(
                                        gt_trans, (0, 15), value=0.0
                                    ).repeat((self._bps_dim, 1)),
                                ],
                                dim=0,
                            ).cpu()
                            # =================================================================

                        with open(sample_pth, "wb") as f:
                            pickle.dump((sample, label), f)
                        sample_paths.append(sample_pth)
                        if visualize and not has_visualized:
                            print("[*] Plotting CHOIR... (please be patient)")
                            visualize_CHOIR(
                                gt_choir.squeeze(0),
                                self._bps,
                                gt_scalar,
                                gt_verts.squeeze(0),
                                gt_anchors.squeeze(0),
                                obj_mesh,
                                obj_ptcld,
                                gt_rescaled_ref_pts.squeeze(0),
                                affine_mano,
                            )
                            faces = affine_mano.faces
                            gt_MANO_mesh = Trimesh(
                                gt_verts.squeeze(0).cpu().numpy(), faces.cpu().numpy()
                            )
                            pred_MANO_mesh = Trimesh(
                                verts.squeeze(0).cpu().numpy(), faces.cpu().numpy()
                            )
                            visualize_MANO(
                                pred_MANO_mesh, obj_mesh=obj_mesh, gt_hand=gt_MANO_mesh
                            )
                            visualize_CHOIR_prediction(
                                gt_choir,
                                gt_choir,
                                self._bps,
                                scalar,
                                gt_scalar,
                                rescaled_ref_pts,
                                gt_rescaled_ref_pts,
                                gt_verts,
                                gt_joints,
                                gt_anchors,
                                is_rhand=(hand_idx == "right"),
                                use_smplx=False,
                                remap_bps_distances=self._remap_bps_distances,
                                exponential_map_w=self._exponential_map_w,
                            )
                            has_visualized = True
                    grasp_paths.append(sample_paths)
                pbar.update()
        return grasp_paths
