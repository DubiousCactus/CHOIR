# vim:fenc=utf-8
#
# Copyright © 2022 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
ContactPose dataset.
"""

import os
import os.path as osp
import pickle
import random
import sys
from contextlib import redirect_stdout
from typing import List, Tuple

import blosc
import numpy as np
import torch
from hydra.utils import get_original_cwd
from open3d import io as o3dio
from pytorch3d.transforms import matrix_to_rotation_6d
from torchmetrics import MeanMetric
from tqdm import tqdm
from trimesh import Trimesh

from conf.project import ANSI_COLORS, Theme
from dataset.base import BaseDataset
from model.affine_mano import AffineMANO
from utils import colorize, to_cuda_
from utils.dataset import augment_hand_object_pose, compute_choir, get_scalar
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
        use_contactopt_splits: bool = False,
        use_improved_contactopt_splits: bool = False,
        validation_objects: int = 3,
        test_objects: int = 2,
        perturbation_level: int = 0,
        obj_ptcld_size: int = 3000,
        bps_dim: int = 1024,
        noisy_samples_per_grasp: int = 100,
        min_views_per_grasp: int = 1,
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
        random_anchor_assignment: bool = False,
        eval_observations_plateau: bool = False,
        eval_anchor_assignment: bool = False,
        use_deltas: bool = False,
        use_bps_grid: bool = False,
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
        self._use_contactopt_splits = use_contactopt_splits
        self._use_improved_contactopt_splits = use_improved_contactopt_splits
        # Using ContactOpt's parameters
        if self._use_contactopt_splits or self._use_improved_contactopt_splits:
            if split == "train":
                noisy_samples_per_grasp = 16
            else:
                noisy_samples_per_grasp = 4

        self._eval_observation_plateau = eval_observations_plateau
        if eval_observations_plateau:
            if split == "test":
                noisy_samples_per_grasp = 15

        if eval_anchor_assignment:
            noisy_samples_per_grasp = 1
            max_views_per_grasp = 1

        self._eval_anchor_assignment = eval_anchor_assignment

        super().__init__(
            dataset_name="ContactPose",
            bps_dim=bps_dim,
            validation_objects=validation_objects,
            test_objects=test_objects,
            right_hand_only=right_hand_only,
            obj_ptcld_size=obj_ptcld_size,
            perturbation_level=perturbation_level,
            min_views_per_grasp=min_views_per_grasp,
            max_views_per_grasp=max_views_per_grasp,
            noisy_samples_per_grasp=noisy_samples_per_grasp,
            rescale=rescale,
            center_on_object_com=center_on_object_com,
            remap_bps_distances=remap_bps_distances,
            exponential_map_w=exponential_map_w,
            random_anchor_assignment=random_anchor_assignment,
            use_deltas=use_deltas,
            use_bps_grid=use_bps_grid,
            augment=augment,
            n_augs=n_augs,
            split=split,
            tiny=tiny,
            seed=seed,
            debug=debug,
        )

    @property
    def eval_anchor_assignment(self):
        return self._eval_anchor_assignment

    @property
    def eval_observation_plateau(self):
        return self._eval_observation_plateau

    @property
    def theta_dim(self):
        return 18

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

        # First, build a dictionary of object names to the participant, intent, and hand used. The
        # reason we don't do it all in one pass is that some participants may not manipulate some
        # objects.
        participant_splits = (
            self._use_contactopt_splits
            or self._use_improved_contactopt_splits
            or self._eval_anchor_assignment
        )
        cp_dataset = {} if not participant_splits else []
        n_participants = 3 if tiny else 51
        for p_num in range(1, n_participants):
            for intent in ["use", "handoff"]:
                for obj_name in get_object_names(p_num, intent):
                    if participant_splits:
                        cp_dataset.append((p_num, intent, obj_name))
                    else:
                        if obj_name not in cp_dataset:
                            cp_dataset[obj_name] = []
                        cp_dataset[obj_name].append((p_num, intent, "right"))
                        if not self._right_hand_only:
                            cp_dataset[obj_name].append((p_num, intent, "left"))
        hand_indices = {"right": 1, "left": 0}

        if participant_splits:
            assert not (
                self._use_contactopt_splits and self._use_improved_contactopt_splits
            ), "You can't use both ContactOpt's splits and the improved splits."
            # Naive split by grasp or almost by participant. Not great, but that's what ContactOpt does.
            low_p, high_p = 0, 0
            if self._use_contactopt_splits:
                if split == "train":
                    low_p, high_p = 0, 0.8
                else:
                    low_p, high_p = 0.8, 1.0
            elif self._use_improved_contactopt_splits:
                # I'm improving it so that we have a validation split here.
                if split == "train":
                    low_p, high_p = 0, 0.7
                elif split == "val":
                    low_p, high_p = 0.7, 0.8
                elif split == "test":
                    low_p, high_p = 0.8, 1.0
            elif self._eval_anchor_assignment:
                if split != "test":
                    low_p, high_p = 0.0, 0.1
                else:
                    low_p, high_p = 0.0, 1.0  # Use all the dataset to test this!

            low_split = int(len(cp_dataset) * low_p)
            high_split = int(len(cp_dataset) * high_p)
            cp_dataset = cp_dataset[
                low_split:high_split
            ]  # [0.0, 0.8] for train, [0.8, 1.0] for test
            # Now, reformat the dataset to be a dictionary of object names to a list of grasps so
            # that the rest of the code works as is (I wrote this code before I knew about the
            # splits in ContactOpt).
            by_object = {}
            for p_num, intent, obj_name in cp_dataset:
                if obj_name not in by_object:
                    by_object[obj_name] = []
                by_object[obj_name].append((p_num, intent, "right"))
                if not self._right_hand_only:
                    by_object[obj_name].append((p_num, intent, "left"))
            cp_dataset = by_object
            object_names = sorted(list(cp_dataset.keys()))
        else:
            # Split by objects
            # Keep only the first n_objects - validation_objects if we're in training mode, and the last
            # validation_objects if we're in eval mode:
            object_names = sorted(list(cp_dataset.keys()))
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

        # Now, build the dataset.
        # objects: unique object paths
        # objects_w_contacts: object with contacts paths
        # grasps: MANO parameters w/ global pose pickle paths
        objects_w_contacts, grasps = [], []
        print(f"[*] Loading ContactPose{' (tiny)' if tiny else ''}...")

        dataset_path = osp.join(
            self._cache_dir,
            f"dataset_{split}_{n_participants}-participants"
            + (
                f"_{self._validation_objects}-val-held-out"
                + f"_{self._test_objects}-test-held-out"
                if not participant_splits
                else f"_from-{low_p}_to-{high_p}_split"
            )
            + f"_{self._obj_ptcld_size}-obj-pts"
            + f"{'right-hand' if self._right_hand_only else 'both-hands'}_seed-{seed}.pkl",
        )
        if osp.isfile(dataset_path):
            with open(dataset_path, "rb") as f:
                compressed_pkl = f.read()
                objects_w_contacts, grasps, n_left, n_right = pickle.loads(
                    blosc.decompress(compressed_pkl)
                )
        else:
            n_left, n_right = 0, 0
            for obj_name in tqdm(object_names):
                for p_num, intent, hand in cp_dataset[obj_name]:
                    with redirect_stdout(None):
                        cp = ContactPose(p_num, intent, obj_name)
                    random_grasp_frame = (
                        cp.mano_params[hand_indices[hand]],
                        cp.mano_meshes()[hand_indices[hand]],
                    )  # {'pose': _, 'betas': _, 'hTm': _} for mano, {'vertices': _, 'faces': _, 'joints': _} for mesh
                    if cp._valid_hands != [1] and hand_indices[hand] == 1:
                        # We want the right hand, but this is anything else than just the right hand
                        continue
                    obj_mesh_w_contacts_path, grasp = (
                        cp.contactmap_filename,
                        random_grasp_frame,
                    )
                    objects_w_contacts.append(obj_mesh_w_contacts_path)
                    grasps.append(
                        {
                            "grasp": grasp,
                            "p_num": p_num,
                            "intent": intent,
                            "obj_name": obj_name,
                            "hand_idx": hand,
                        }
                    )
                    n_left += 1 if hand == "left" else 0
                    n_right += 1 if hand == "right" else 0
            with open(dataset_path, "wb") as f:
                pkl = pickle.dumps((objects_w_contacts, grasps, n_left, n_right))
                compressed_pkl = blosc.compress(pkl)
                f.write(compressed_pkl)

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
            # f"dataset_{hashlib.shake_256(dataset_name.encode()).hexdigest(8)}_"
            f"perturbed-{self._perturbation_level}_"
            + f"_{self._obj_ptcld_size}-obj-pts"
            + f"_{'right-hand' if self._right_hand_only else 'both-hands'}"
            + f"_{self._bps_dim}-bps_"
            if self._use_bps_grid
            else "bps-grid_"
            + f"{'_object-centered' if self._center_on_object_com else ''}"
            + f"_{self._rescale}-rescaled"
            + f"{'_exponential_mapped' if self._remap_bps_distances else ''}"
            + (f"-{self._exponential_map_w}" if self._remap_bps_distances else "")
            + f"_{'random-anchors' if self._random_anchor_assignment else 'ordered-anchors'}"
            + f"{'_deltas' if self._use_deltas else ''}"
            + f"_{split}{'-augmented' if self._augment else ''}",
        )
        if not osp.isdir(samples_labels_pickle_pth):
            os.makedirs(samples_labels_pickle_pth)
        affine_mano: AffineMANO = to_cuda_(AffineMANO(for_contactpose=True))  # type: ignore

        n_augs = self._n_augs if self._augment else 0

        # For each object-grasp pair, compute the CHOIR field.
        print("[*] Computing CHOIR fields...")
        pbar = tqdm(total=len(objects_w_contacts) * (n_augs + 1))
        dataset_mpjpe = MeanMetric()
        dataset_root_aligned_mpjpe = MeanMetric()
        for mesh_pth, grasp_data in zip(objects_w_contacts, grasps):
            """
            grasp_data = {
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
            # TODO: Compute Procrustes Aligned grasp, then procrustes distances for the whole
            # dataset. Each augmented grasp is still the same shape. And we'll maintain that for
            # perturbed grasps as well so that the model learns the underlying shape similarities
            # even with noise.
            for k in range(n_augs + 1):
                grasp_name = f"{obj_name}_{p_num}_{intent}_{hand_idx}_aug-{k}"
                if not osp.isdir(osp.join(samples_labels_pickle_pth, grasp_name)):
                    os.makedirs(osp.join(samples_labels_pickle_pth, grasp_name))
                if (
                    len(os.listdir(osp.join(samples_labels_pickle_pth, grasp_name)))
                    >= self._seq_len
                ):
                    grasp_paths.append(
                        [
                            osp.join(
                                samples_labels_pickle_pth,
                                grasp_name,
                                f"sample_{i:06d}.pkl",
                            )
                            for i in range(self._seq_len)
                        ]
                    )
                else:
                    visualize = self._debug and (random.Random().random() < 0.05)
                    has_visualized = False
                    # ================== Original Hand-Object Pair ==================
                    mano_params = grasp_data["grasp"][0].copy()
                    gt_hTm = (
                        torch.from_numpy(mano_params["hTm"]).float().unsqueeze(0).cuda()
                    )
                    obj_mesh = o3dio.read_triangle_mesh(mesh_pth)
                    # =================== Apply augmentation =========================
                    if self._augment and k > 0:
                        obj_mesh, gt_hTm = augment_hand_object_pose(
                            obj_mesh, gt_hTm, around_z=False
                        )
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
                    # When we augment we necessarily recenter on the object, so we don't need to do
                    # it here (except for k=0).
                    if self._center_on_object_com and not (self._augment and k > 0):
                        obj_center = torch.from_numpy(obj_mesh.get_center())
                        obj_mesh.translate(-obj_center)
                        obj_ptcld -= obj_center.to(obj_ptcld.device)
                        gt_trans -= obj_center.to(gt_trans.device)
                    # ================ Compute GT anchors and verts ==================
                    gt_verts, gt_joints = affine_mano(
                        gt_theta, gt_beta, gt_rot_6d, gt_trans
                    )
                    gt_anchors = affine_mano.get_anchors(gt_verts)
                    # ================== Rescaled Hand-Object Pair ==================
                    gt_scalar = get_scalar(gt_anchors, obj_ptcld, self._rescale)

                    # I know it's bad to do CUDA stuff in the dataset if I want to use multiple
                    # workers, but bps_torch is forcing my hand here so I might as well help it.
                    gt_choir, gt_rescaled_ref_pts = compute_choir(
                        to_cuda_(obj_ptcld).unsqueeze(0),
                        to_cuda_(gt_anchors),
                        scalar=gt_scalar,
                        bps=to_cuda_(self._bps).unsqueeze(0),  # type: ignore
                        anchor_indices=self._anchor_indices.cuda(),  # type: ignore
                        remap_bps_distances=self._remap_bps_distances,
                        exponential_map_w=self._exponential_map_w,
                        use_deltas=self._use_deltas,
                    )

                    sample_paths = []
                    # ================= Perturbed Hand-Object pairs =================
                    for j in range(self._seq_len):
                        sample_pth = osp.join(
                            samples_labels_pickle_pth, grasp_name, f"sample_{j:06d}.pkl"
                        )
                        if osp.isfile(sample_pth):
                            sample_paths.append(sample_pth)
                            continue

                        theta, beta, rot_6d, trans = (
                            gt_theta.clone(),
                            gt_beta.clone(),
                            gt_rot_6d.clone(),
                            gt_trans.clone(),
                        )
                        trans_noise = (
                            torch.randn(3, device=trans.device)
                            * self._perturbations[self._perturbation_level]["trans"]
                        )
                        pose_noise = torch.cat(
                            [
                                torch.randn(3, device=theta.device)
                                * self._perturbations[self._perturbation_level]["rot"],
                                torch.randn(15, device=theta.device)
                                * self._perturbations[self._perturbation_level]["pca"],
                            ]
                        )
                        theta += pose_noise
                        trans += trans_noise

                        verts, joints = affine_mano(theta, beta, rot_6d, trans)

                        mpjpe = torch.linalg.vector_norm(
                            joints.squeeze(0) - gt_joints.squeeze(0), dim=1, ord=2
                        ).mean()
                        root_aligned_mpjpe = torch.linalg.vector_norm(
                            (joints - joints[:, 0:1, :]).squeeze(0)
                            - (gt_joints - gt_joints[:, 0:1, :]).squeeze(0),
                            dim=1,
                            ord=2,
                        ).mean()
                        dataset_mpjpe.update(mpjpe.item())
                        dataset_root_aligned_mpjpe.update(root_aligned_mpjpe.item())

                        anchors = affine_mano.get_anchors(verts)
                        scalar = get_scalar(anchors, obj_ptcld, self._rescale)
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
                            use_deltas=self._use_deltas,
                        )
                        sample, label = (
                            choir.squeeze().cpu().numpy(),
                            rescaled_ref_pts.squeeze().cpu().numpy(),
                            scalar.cpu().numpy(),
                            np.ones((1)) if hand_idx == "right" else np.zeros((1)),
                            theta.squeeze().cpu().numpy(),
                            beta.squeeze().cpu().numpy(),
                            rot_6d.squeeze().cpu().numpy(),
                            trans.squeeze().cpu().numpy(),
                        ), (
                            gt_choir.squeeze().cpu().numpy(),
                            gt_rescaled_ref_pts.squeeze().cpu().numpy(),
                            gt_scalar.cpu().numpy(),
                            gt_joints.squeeze().cpu().numpy(),
                            gt_anchors.squeeze().cpu().numpy(),
                            gt_theta.squeeze().cpu().numpy(),
                            gt_beta.squeeze().cpu().numpy(),
                            gt_rot_6d.squeeze().cpu().numpy(),
                            gt_trans.squeeze().cpu().numpy(),
                        )
                        # =================================================================

                        with open(sample_pth, "wb") as f:
                            pkl = pickle.dumps((sample, label, mesh_pth))
                            compressed_pkl = blosc.compress(pkl)
                            f.write(compressed_pkl)
                        sample_paths.append(sample_pth)

                        if visualize and not has_visualized:
                            print(
                                f"[*] Plotting CHOIR for {grasp_name} ... (please be patient)"
                            )
                            visualize_CHOIR(
                                # choir.squeeze(0),
                                gt_choir.squeeze(0),
                                self._bps,
                                self._anchor_indices,
                                scalar,
                                gt_verts.squeeze(0),
                                gt_anchors.squeeze(0),
                                obj_mesh,
                                obj_ptcld,
                                gt_rescaled_ref_pts.squeeze(0),
                                affine_mano,
                                use_deltas=self._use_deltas,
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
                                self._anchor_indices,
                                scalar,
                                gt_scalar,
                                rescaled_ref_pts,
                                gt_rescaled_ref_pts,
                                gt_verts,
                                gt_joints,
                                gt_anchors,
                                is_rhand=(hand_idx == "right"),
                                use_smplx=False,
                                dataset="ContactPose",
                                remap_bps_distances=self._remap_bps_distances,
                                exponential_map_w=self._exponential_map_w,
                                use_deltas=self._use_deltas,
                            )
                            has_visualized = True
                    grasp_paths.append(sample_paths)
                pbar.update()
        print(
            f"[*] Dataset MPJPE (mm): {dataset_mpjpe.compute().item() * self.base_unit}"
        )
        print(
            f"[*] Dataset Root-aligned MPJPE (mm): {dataset_root_aligned_mpjpe.compute().item() * self.base_unit}"
        )
        return grasp_paths
