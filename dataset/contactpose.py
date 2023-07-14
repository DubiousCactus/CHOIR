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
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from bps_torch.tools import sample_sphere_uniform
from hydra.utils import get_original_cwd
from manotorch.anchorlayer import AnchorLayer
from open3d import io as o3dio
from pytorch3d.structures import Pointclouds
from pytorch3d.transforms import matrix_to_rotation_6d
from tqdm import tqdm

from conf.project import ANSI_COLORS, Theme
from dataset.base import BaseDataset
from model.affine_mano import AffineMANO
from utils import colorize, to_cuda_
from utils.dataset import compute_choir
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
        n_random_choir_per_sample: int = 1000,
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
        self._n_random_choir_per_sample = (
            n_random_choir_per_sample if anchor_assignment == "random" else 1
        )
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
            scaling=scaling,
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

    @property
    def anchor_assignment(self) -> str:
        return self._anchor_assignment

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
            + f"{self._anchor_assignment}-assigned_"
            + f"{self._bps_dim}-bps_"
            + f"{split}",
        )
        if not osp.isdir(samples_labels_pickle_pth):
            os.makedirs(samples_labels_pickle_pth)
        affine_mano: AffineMANO = to_cuda_(AffineMANO())  # type: ignore
        anchor_layer = AnchorLayer(anchor_root="vendor/manotorch/assets/anchor").cuda()
        hand_object_stats_pth = osp.join(
            samples_labels_pickle_pth,
            f"hand_object_stats_{self._obj_ptcld_size}-pts.pkl",
        )
        hand_object_scalars = None
        if osp.isfile(hand_object_stats_pth):
            with open(hand_object_stats_pth, "rb") as f:
                hand_object_scalars = pickle.load(f)
        else:
            print("[*] Computing hand-object statistics...")
            # TODO: This is very expensive in memory so we should batch it.
            hand_object_pointclouds = []
            for objects_w_contacts_pth, grasp_pth in tqdm(
                zip(objects_w_contacts, grasps), total=len(objects_w_contacts)
            ):
                with open(grasp_pth, "rb") as f:
                    grasp_data = pickle.load(f)
                # obj_ptcld = pointclouds[objects.index(grasp_data["obj_name"])]
                obj_contacts_mesh = o3dio.read_triangle_mesh(objects_w_contacts_pth)
                obj_pointcloud = torch.from_numpy(
                    np.asarray(
                        obj_contacts_mesh.sample_points_uniformly(self._obj_ptcld_size).points  # type: ignore
                    )
                ).cpu()
                # get the anchors instead of a hand point cloud:
                mano_params = grasp_data["grasp"][0]
                hTm = torch.from_numpy(mano_params["hTm"]).float().unsqueeze(0).cuda()
                rot_6d = matrix_to_rotation_6d(hTm[:, :3, :3])
                trans = hTm[:, :3, 3]
                verts, _ = affine_mano(
                    torch.tensor(mano_params["pose"]).unsqueeze(0).cuda(),
                    torch.tensor(mano_params["betas"]).unsqueeze(0).cuda(),
                    rot_6d,
                    trans,
                )
                anchors = anchor_layer(verts)
                # Build a pointcloud from the object points + hand anchors with open3D/Pytorch3D and compute
                # the mean and std of all samples.
                hand_object_pointclouds.append(
                    torch.cat([obj_pointcloud, anchors.squeeze(0).cpu()], dim=0)
                )
            hand_object_pointclouds = Pointclouds(
                torch.stack(hand_object_pointclouds, dim=0)
            )
            # Now scale the pointclouds to unit sphere. First, find the scalar to apply to each
            # pointcloud. We can get their bounding boxes and find the min and max of the
            # diagonal of the bounding boxes, and then compute each scalar such that the
            # diagonal of the bounding box is 1.

            # (N, 3, 2) where bbox[i,j] gives the min and max values of mesh i along axis j. So
            # bbox[i,:,0] is the min values of mesh i along all axes, and bbox[i,:,1] is the max
            # values of mesh i along all axes.
            hand_object_bboxes = hand_object_pointclouds.get_bounding_boxes()
            hand_object_bboxes_diag = torch.norm(
                hand_object_bboxes[:, :, 1] - hand_object_bboxes[:, :, 0], dim=1
            )  # (N,)
            hand_object_scalars = 1.0 / hand_object_bboxes_diag  # (N,)
            # Make sure that the new bounding boxes have a diagonal of 1.
            rescaled_hand_object_pointclouds = hand_object_pointclouds.scale(
                hand_object_scalars.unsqueeze(1)
            )  # (N, 3, M)
            hand_object_bboxes = rescaled_hand_object_pointclouds.get_bounding_boxes()
            hand_object_bboxes_diag = torch.norm(
                hand_object_bboxes[:, :, 1] - hand_object_bboxes[:, :, 0], dim=1
            )  # (N,)
            assert torch.allclose(
                hand_object_bboxes_diag, torch.ones_like(hand_object_bboxes_diag)
            ), "Bounding boxes are not unit cubes."
            with open(hand_object_stats_pth, "wb") as f:
                pickle.dump(
                    hand_object_scalars.cpu().numpy().tolist(),
                    f,
                )

        # For each object-grasp pair, compute the CHOIR field.
        print("[*] Computing CHOIR fields...")
        for scalar, mesh_pth, grasp_pth in tqdm(
            zip(hand_object_scalars, objects_w_contacts, grasps),
            total=len(objects_w_contacts),
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
                >= self._n_random_choir_per_sample
            ):
                sample_paths += [
                    osp.join(
                        samples_labels_pickle_pth, grasp_name, f"sample_{i:06d}.pkl"
                    )
                    for i in range(self._n_random_choir_per_sample)
                ]
            else:
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
                obj_mesh = o3dio.read_triangle_mesh(mesh_pth)
                obj_ptcld = torch.from_numpy(
                    np.asarray(
                        obj_mesh.sample_points_uniformly(self._obj_ptcld_size).points  # type: ignore
                    )
                ).cpu()
                visualize = self._debug and (random.random() < 0.1)
                has_visualized = False
                for j in range(self._n_random_choir_per_sample):
                    sample_pth = osp.join(
                        samples_labels_pickle_pth, grasp_name, f"sample_{j:06d}.pkl"
                    )
                    if osp.isfile(sample_pth):
                        sample_paths.append(sample_pth)
                        continue
                    anchors = anchor_layer(verts)
                    # I know it's bad to do CUDA stuff in the dataset if I want to use multiple
                    # workers, but bps_torch is forcing my hand here so I might as well help it.
                    choir, rescaled_ref_pts = compute_choir(
                        to_cuda_(obj_ptcld).unsqueeze(0),
                        to_cuda_(anchors),
                        scalar=scalar,
                        bps=to_cuda_(self._bps).unsqueeze(0),  # type: ignore
                        anchor_assignment=self._anchor_assignment,
                    )
                    # anchor_orientations = torch.nn.functional.normalize(
                    # anchor_deltas, dim=1
                    # )
                    # Compute the dense MANO contact map
                    # hand_contacts = compute_hand_contacts_simple(
                    # ref_pts.float(), verts.float()
                    # )

                    # Remove batch dimension
                    choir = choir.squeeze(0).cpu()
                    # pcl_mean = pcl_mean.squeeze(0).cpu()
                    # pcl_scalar = pcl_scalar.squeeze(0).cpu()
                    # hand_contacts = hand_contacts.squeeze(0).cpu()
                    joints = joints.squeeze(0).cpu()
                    # anchor_orientations = anchor_orientations.squeeze(0).cpu()
                    anchors = anchors.squeeze(0).cpu()
                    rot_6d = rot_6d.squeeze(0).cpu()
                    trans = trans.squeeze(0).cpu()
                    rescaled_ref_pts = rescaled_ref_pts.squeeze(0).cpu()

                    sample = (
                        choir,
                        # hand_contacts,
                        # pcl_mean,
                        # pcl_scalar,
                        # self._bps_dim,
                        rescaled_ref_pts,
                        scalar,
                    )
                    label = (
                        choir,
                        # hand_contacts,
                        # anchor_orientations,
                        joints,
                        anchors,
                        torch.tensor(mano_params["pose"]).cpu(),
                        torch.tensor(mano_params["betas"]).cpu(),
                        rot_6d,
                        trans,
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
                        mesh = o3dio.read_triangle_mesh(mesh_pth)
                        visualize_CHOIR(
                            choir,
                            self._bps,
                            scalar,
                            # hand_contacts,
                            verts,
                            anchors,
                            # anchor_orientations,
                            mesh,
                            obj_ptcld,
                            rescaled_ref_pts.squeeze(0),
                            affine_mano,
                            self._anchor_assignment,
                        )
                        has_visualized = True
        return sample_paths
