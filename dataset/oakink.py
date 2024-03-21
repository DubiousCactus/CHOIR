#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
OakInk shape dataset. Each sample is a tuple of (D_train, D_test) sampled from an UDF generated
from the corresponding OakInk shape. D_train is a sparse (un)signed distance field sampled in a uniform
ball. D_test is a dense (un)signed distance field sampled in a uniform ball.
"""

import os
import os.path as osp
import pickle
import random
from typing import List, Tuple

import blosc2
import numpy as np
import open3d
import open3d.geometry as o3dg
import open3d.utility as o3du
import torch
from pytorch3d.transforms.rotation_conversions import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
)
from torchmetrics import MeanMetric
from tqdm import tqdm
from trimesh import Trimesh

from dataset.base import BaseDataset
from globals import FLAT_LOWER_INDICES
from model.affine_mano import AffineMANO
from utils import debug_methods
from utils.dataset import (
    augment_hand_object_pose,
    compute_anchor_gaussians,
    compute_choir,
    get_contact_counts_by_neighbourhoods,
    get_scalar,
)
from utils.visualization import (
    visualize_3D_gaussians_on_hand_mesh,
    visualize_hand_contacts_from_3D_gaussians,
    visualize_MANO,
)


@debug_methods
class OakInkDataset(BaseDataset):
    base_unit = 1000.0  # The dataset is in meters, we want to work in mm.

    # ====== UDFs ======
    gt_udf_mean = torch.tensor([0.4945, 0.3616])  # object distances, hand distances
    gt_udf_std = torch.tensor([0.1532, 0.1297])  # object distances, hand distances
    noisy_udf_mean = torch.tensor([0.4945, 0.3409])  # object distances, hand distances
    noisy_udf_std = torch.tensor([0.1532, 0.1399])  # object distances, hand distances

    # ====== Keypoints ======
    gt_kp_obj_mean, gt_kp_hand_mean = torch.tensor(
        [8.6208e-05, -1.5328e-04, -1.1284e-0]
    ), torch.tensor([-0.0005, -0.0014, -0.0169])
    gt_kp_obj_std, gt_kp_hand_std = torch.tensor(
        [0.0318, 0.0310, 0.0690]
    ), torch.tensor([0.0441, 0.0448, 0.0549])
    noisy_kp_obj_mean, noisy_kp_hand_mean = torch.tensor(
        [8.6208e-05, -1.5328e-04, -1.1284e-02]
    ), torch.tensor([-0.0008, -0.0017, -0.0176])
    noisy_kp_obj_std, noisy_kp_hand_std = torch.tensor(
        [0.0318, 0.0310, 0.0690]
    ), torch.tensor([0.0706, 0.0706, 0.0783])

    # ====== Contacts ======
    contacts_mean = torch.tensor(
        [
            -1.0005e-05,  # mean x
            -3.7385e-05,  # mean y
            1.2783e-04,  # mean z
            5.6400e-04,  # cholesky-decomped cov 00
            5.2135e-06,  # cholesky-decomped cov 03
            4.2053e-04,  # cholesky-decomped cov 04
            8.6100e-08,  # cholesky-decomped cov 06
            4.7047e-07,  # cholesky-decomped cov 07
            4.4317e-04,  # cholesky-decomped cov 08
        ]
    )
    contacts_std = torch.tensor(
        [
            0.0013,  # mean x
            0.0013,  # mean y
            0.0012,  # mean z
            0.0013,  # cholesky-decomped cov 00
            0.0009,  # cholesky-decomped cov 03
            0.0010,  # cholesky-decomped cov 04
            0.0007,  # cholesky-decomped cov 06
            0.0008,  # cholesky-decomped cov 07
            0.0010,  # cholesky-decomped cov 08
        ]
    )

    contacts_min = torch.tensor(
        [-0.0099, -0.0100, -0.0100, 0.0000, -0.0145, 0.0000, -0.0147, -0.0130, 0.0000]
    )
    contacts_max = torch.tensor(
        [0.0100, 0.0100, 0.0100, 0.0157, 0.0147, 0.0147, 0.0145, 0.0126, 0.0135]
    )

    def __init__(
        self,
        dataset_root: str,
        split: str,
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
        model_contacts: bool = False,
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
            {"trans": 0.0, "rot": 0.0, "axisangle": 0.0},  # Level 0
            {"trans": 0.02, "rot": 0.05, "axisangle": 0.3},  # Level 1
            {
                "trans": 0.05,
                "rot": 0.15,  # wrist
                "axisangle": 0.3,  # fingers
            },  # Level 2 (0.05m, 0.15 axis-angle in radians, 0.2 axis-angle in radians)
            {
                "trans": 0.005,
                "rot": 0.05,
                "axisangle": 0.3,
            },  # Level 3 (0.005m, 0.15 axis-angle in radians, 0.3 axis-angle in radians), for tests with TTO
        ]

        if eval_anchor_assignment:
            noisy_samples_per_grasp = 1
            max_views_per_grasp = 1

        self._eval_anchor_assignment = eval_anchor_assignment
        self._model_contacts = model_contacts
        self._dataset_root = dataset_root

        super().__init__(
            dataset_name="OakInk",
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
    def theta_dim(self):
        return 48

    def _load_objects_and_grasps(
        self, tiny: bool, split: str, seed: int = 0
    ) -> Tuple[dict, list, list, str]:
        idx = 0
        if tiny:
            n_samples = 10000 if split == "train" else 2000
        else:
            os.environ["OAKINK_DIR"] = self._dataset_root
            from oikit.oi_shape import OakInkShape

            dataset = OakInkShape(
                data_split=split, mano_assets_root="vendor/manotorch/assets/mano"
            )
            n_samples = (
                len(dataset) if not tiny else (10000 if split == "train" else 2000)
            )
        dataset_path = osp.join(
            self._cache_dir,
            f"{split}_{n_samples}-samples" + f"{'_tiny' if tiny else ''}.pkl",
        )
        obj_and_grasps_path = osp.join(
            self._cache_dir,
            f"{split}_{n_samples}-samples" + f"{'_tiny' if tiny else ''}",
        )
        os.makedirs(obj_and_grasps_path, exist_ok=True)
        object_ids, objects, grasps = [], [], []
        print(f"[*] Loading OakInk{' (tiny)' if tiny else ''}...")
        if osp.isfile(dataset_path):
            with open(dataset_path, "rb") as f:
                compressed_pkl = f.read()
                object_ids, objects, grasps = pickle.loads(
                    blosc2.decompress(compressed_pkl)
                )
            # new_objects, new_grasps = [], []
            # for obj_path, grasp_path in zip(objects, grasps):
            # obj_path = obj_path.replace("/media/data2/moralest/", "/Users/cactus/Code/")
            # grasp_path = grasp_path.replace(
            # "/media/data2/moralest/", "/Users/cactus/Code/"
            # )
            # new_objects.append(obj_path)
            # new_grasps.append(grasp_path)
            # with open(dataset_path, "wb") as f:
            # pkl = pickle.dumps((new_objects, new_grasps))
            # compressed_pkl = blosc2.compress(pkl)
            # f.write(compressed_pkl)
        else:
            os.environ["OAKINK_DIR"] = self._dataset_root
            from oikit.oi_shape import OakInkShape

            pbar = tqdm(total=n_samples)
            for shape in OakInkShape(
                data_split=split, mano_assets_root="vendor/manotorch/assets/mano"
            ):
                if tiny and idx == n_samples:
                    break
                """
                Annotations: "Object's .obj models in its canonical system; MANO's pose & shape parameters
                and vertex 3D locations in object's canonical system."

                shape has the following structure:
                {
                    seq_id: str
                    obj_id: str
                    joints: 21x3
                    verts: 778x3
                    hand_pose: axis-angle pose for MANO, in obj-canonical space
                    hand_shape: beta for MANO, (10,)
                    hand_tsl: translation in obj-canonical space for MANO, (3,). This is actually
                                the translation for CENTER JOINT, not necessarily the wrist. It's
                                actually joint 9.
                    is_virtual: bool
                    raw_obj_id: str
                    action_id: one of 0001, 0002, 0003, 0004 (use, hold, lift-up, handover).
                    subject_id: int
                    subject_alt_id: int
                    seq_ts: str (Date and time)
                    source: pickle file path for the source data.
                    pass_stage: str, one of ['pass1', ...?]
                    alt_grasp_item: {'alt_joints': ..., 'alt_verts': ..., 'alt_hand_pose': ..., 'alt_hand_shape': ..., 'alt_hand_tsl': ...}
                    obj_verts: (N, 3)
                    obj_faces: (M, 3)

                 ----- And optionally, for hand-over: -----
                    alt_joints: (21, 3)
                    alt_verts: (778, 3)
                    alt_hand_pose: (48,)
                    alt_hand_shape: (10,)
                    alt_hand_tsl: (3,)
                }
                """
                obj_path = osp.join(obj_and_grasps_path, f"{shape['obj_id']}.pkl")
                grasp_path = osp.join(
                    obj_and_grasps_path, f"grasp_{idx}_{shape['seq_id']}.pkl"
                )

                if not osp.isfile(obj_path):
                    with open(obj_path, "wb") as f:
                        pickle.dump(
                            {
                                "verts": shape["obj_verts"],
                                "faces": shape["obj_faces"],
                            },
                            f,
                        )
                if not osp.isfile(grasp_path):
                    with open(grasp_path, "wb") as f:
                        pickle.dump(
                            {
                                "seq_id": shape["seq_id"],
                                "obj_id": shape["obj_id"],
                                "action_id": shape["action_id"],
                                "joints": shape["joints"],
                                "verts": shape["verts"],
                                "hand_pose": shape["hand_pose"],
                                "hand_shape": shape["hand_shape"],
                                "hand_tsl": shape["hand_tsl"],
                            },
                            f,
                        )

                objects.append(obj_path)
                object_ids.append(shape["obj_id"])
                grasps.append(grasp_path)
                idx += 1
                pbar.update()
            with open(dataset_path, "wb") as f:
                pkl = pickle.dumps((object_ids, objects, grasps))
                compressed_pkl = blosc2.compress(pkl)
                f.write(compressed_pkl)
        print(f"[*] Loaded {len(set(object_ids))} objects and {len(grasps)} grasps.")
        assert len(objects) == len(grasps)
        return (
            objects,
            grasps,
            osp.basename(dataset_path.split(".")[0]),
        )

    def _load(
        self,
        split: str,
        objects: List[str],
        grasps: List,
        dataset_name: str,
    ) -> List[str]:
        samples_labels_pickle_pth = osp.join(
            self._cache_dir,
            "samples_and_labels",
            f"perturbed-{self._perturbation_level}_"
            + f"_{self._obj_ptcld_size}-obj-pts"
            + f"_{self._bps_dim}-bps"
            + (f"-grid" if self._use_bps_grid else "")
            + f"{'_object-centered' if self._center_on_object_com else ''}"
            + f"_{self._rescale}-rescaled"
            + f"{'_exponential_mapped' if self._remap_bps_distances else ''}"
            + (f"-{self._exponential_map_w}" if self._remap_bps_distances else "")
            + f"_{'random-anchors' if self._random_anchor_assignment else 'ordered-anchors'}"
            + f"{'_deltas' if self._use_deltas else ''}"
            + f"{'_contact-gaussians' if self._model_contacts else ''}"
            + f"_{split}{'-augmented' if self._augment else ''}",
        )
        if not osp.isdir(samples_labels_pickle_pth):
            os.makedirs(samples_labels_pickle_pth)
        affine_mano: AffineMANO = AffineMANO(for_oakink=True)  # type: ignore

        n_augs = self._n_augs if self._augment else 0

        # For each object-grasp pair, compute the CHOIR field.
        print("[*] Computing CHOIR fields...")
        grasp_paths = []
        pbar = tqdm(total=len(objects) * (n_augs + 1))
        dataset_mpjpe = MeanMetric()
        dataset_root_aligned_mpjpe = MeanMetric()
        computed = False
        hand_idx = 0
        for obj_pth, grasp_pth in zip(objects, grasps):
            # obj_pth = obj_pth.replace("/media/data3/moralest/", "/Users/cactus/Code/")
            # grasp_pth = grasp_pth.replace(
            # "/media/data3/moralest/", "/Users/cactus/Code/"
            # )
            with open(grasp_pth, "rb") as f:
                grasp = pickle.load(f)
            with open(obj_pth, "rb") as f:
                obj = pickle.load(f)
            obj_id, seq_id, action_id = (
                grasp["obj_id"],
                grasp["seq_id"],
                grasp["action_id"],
            )
            for k in range(n_augs + 1):
                grasp_name = f"{obj_id}_{seq_id}_{action_id}_aug-{k}"
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
                    visualize = self._debug and (random.Random().random() < 0.5)
                    has_visualized = False
                    computed = True
                    # ================== Original Hand-Object Pair ==================
                    obj_mesh = o3dg.TriangleMesh()
                    obj_mesh.vertices = o3du.Vector3dVector(obj["verts"])
                    obj_mesh.triangles = o3du.Vector3iVector(obj["faces"])
                    gt_verts, gt_joints = (
                        torch.from_numpy(grasp["verts"])[None, ...],
                        torch.from_numpy(grasp["joints"])[None, ...],
                    )
                    gt_pose, gt_shape, gt_trans = (
                        torch.from_numpy(grasp["hand_pose"].copy())[None, ...],
                        torch.from_numpy(grasp["hand_shape"].copy())[None, ...],
                        torch.from_numpy(grasp["hand_tsl"].copy())[None, ...],
                    )
                    # print("ORIGINAL")
                    # print(gt_verts.shape, gt_joints.shape)
                    # hand_tmesh = Trimesh(
                    # vertices=gt_verts, faces=affine_mano.faces.cpu().numpy()
                    # )
                    # # Plot the axes gizmo:
                    # pyvista.plot([obj_mesh, hand_tmesh])
                    # =================== Apply augmentation =========================
                    if self._augment and k > 0:
                        # Let's build a 4x4 homogeneous matrix from the MANO root rotation
                        # (axis-angle quaternion) and translation (3D vector):
                        hTm = torch.eye(4)[None, ...]
                        root_rot_quat = gt_pose[:, :3]
                        hTm[:, :3, :3] = axis_angle_to_matrix(root_rot_quat)
                        hTm[:, :3, 3] = gt_trans
                        obj_mesh, hTm = augment_hand_object_pose(
                            obj_mesh, hTm, around_z=False
                        )
                        gt_pose[:, :3] = matrix_to_axis_angle(hTm[:, :3, :3])
                        gt_trans = hTm[:, :3, 3]
                        gt_verts, gt_joints = affine_mano(gt_pose, gt_shape, gt_trans)
                        # print("AUGMENTED")
                        # hand_tmesh = Trimesh(
                        # vertices=gt_verts[0].cpu().numpy(),
                        # faces=affine_mano.faces.cpu().numpy(),
                        # )
                        # pyvista.plot([obj_mesh, hand_tmesh])
                    # =================================================================
                    # gt_verts, gt_joints = affine_mano(
                    # torch.from_numpy(gt_pose).float().unsqueeze(0),
                    # torch.from_numpy(gt_shape).float().unsqueeze(0),
                    # torch.from_numpy(gt_trans).float().unsqueeze(0),
                    # )
                    # print("RECOVERED FROM MANO")
                    # hand_tmesh = Trimesh(
                    # vertices=gt_verts[0].cpu().numpy(),
                    # faces=affine_mano.faces.cpu().numpy(),
                    # )
                    # pyvista.plot([obj_mesh, hand_tmesh])
                    obj_ptcld = torch.from_numpy(
                        np.asarray(
                            obj_mesh.sample_points_uniformly(self._obj_ptcld_size).points  # type: ignore
                        )
                    )
                    # ============ Shift the pair to the object's center ============
                    # When we augment we necessarily recenter on the object, so we don't need to do
                    # it here (except for k=0).
                    if self._center_on_object_com and not (self._augment and k > 0):
                        obj_center = torch.from_numpy(obj_mesh.get_center())
                        obj_mesh.translate(-obj_center)
                        obj_ptcld -= obj_center.to(
                            obj_ptcld.device, dtype=torch.float32
                        )
                        gt_trans -= obj_center.to(gt_trans.device, dtype=torch.float32)
                    # ================ Compute GT anchors ==================
                    gt_anchors = affine_mano.get_anchors(gt_verts)
                    # ================== Rescaled Hand-Object Pair ==================
                    gt_scalar = get_scalar(gt_anchors, obj_ptcld, self._rescale)

                    # I know it's bad to do CUDA stuff in the dataset if I want to use multiple
                    # workers, but bps_torch is forcing my hand here so I might as well help it.
                    gt_choir, gt_rescaled_ref_pts = compute_choir(
                        obj_ptcld.unsqueeze(0),
                        gt_anchors,
                        scalar=gt_scalar,
                        bps=self._bps.unsqueeze(0),  # type: ignore
                        anchor_indices=self._anchor_indices,  # type: ignore
                        remap_bps_distances=self._remap_bps_distances,
                        exponential_map_w=self._exponential_map_w,
                        use_deltas=self._use_deltas,
                    )
                    # ========== Compute ground-truth contact points ================
                    gt_hand_mesh = o3dg.TriangleMesh()
                    gt_hand_mesh.vertices = o3du.Vector3dVector(
                        gt_verts[0].detach().cpu().numpy()
                    )
                    gt_hand_mesh.triangles = o3du.Vector3iVector(
                        affine_mano.faces.detach().cpu().numpy()
                    )
                    gt_hand_mesh.compute_vertex_normals()
                    normals = np.asarray(gt_hand_mesh.vertex_normals)
                    if self._model_contacts:
                        gt_vertex_contacts = get_contact_counts_by_neighbourhoods(
                            gt_verts[0],
                            normals,
                            obj_ptcld,
                            tolerance_cone_angle=4,
                            tolerance_mm=4,
                            base_unit=self.base_unit,
                            K=300,
                        )
                        # ======== Compute ground-truth Contact Gaussians ==============
                        gt_contact_gaussian_params, _ = compute_anchor_gaussians(
                            gt_verts[0],
                            gt_anchors[0],
                            gt_vertex_contacts,
                            base_unit=self.base_unit,
                            anchor_mean_threshold_mm=10,
                            min_contact_points_for_neighbourhood=4,
                        )
                        # TODO: Refactor so that compute_choir is called "compute_bps_rep" or something
                        # of the sort. Cause now I'm augmenting CHOIR to also have Contact Gaussian
                        # parameters! Anyway this is a mess. I think it's easier to do it all in
                        # compute_choir (cause of the anchor indexing stuff).
                        # TODO: I repeat: I will write this in the messiest way so I can get a PoC
                        # ready ASAP. Then I'll refactor it and it'll be so clean you'll never know
                        # this horror happened. (Right?)
                        mu = gt_contact_gaussian_params[
                            :, :3
                        ]  # TODO: Make this scaling cleaner (idk how yet)
                        cov = gt_contact_gaussian_params[:, 3:].reshape(
                            -1, 3, 3
                        )  # TODO: Make this scaling cleaner
                        cholesky_cov = torch.zeros_like(cov)
                        for i in range(cov.shape[0]):
                            # TODO: Work out the minimum threshold by looking through the dataset
                            # with the lowest covariance. Use that for generation. Most likely
                            # non-diagonal elements should be at > thresh and diag elements > 1e-8.
                            if torch.all(cov[i] == 0):
                                continue
                            try:
                                cholesky_cov[i] = torch.linalg.cholesky(cov[i])
                            except torch._C._LinAlgError:
                                nugget = torch.eye(3) * 1e-8
                                cholesky_cov[i] = torch.linalg.cholesky(cov[i] + nugget)
                        lower_tril_cov = cholesky_cov.view(-1, 9)[:, FLAT_LOWER_INDICES]
                        # Basic test:
                        # TODO: Use lower_tril_cholesky_to_covmat()
                        test_lower_tril_mat = torch.zeros_like(cov).view(-1, 9)
                        test_lower_tril_mat[:, FLAT_LOWER_INDICES] = lower_tril_cov
                        test_lower_tril_mat = test_lower_tril_mat.view(-1, 3, 3)
                        approx_cov = torch.einsum(
                            "bik,bjk->bij", test_lower_tril_mat, test_lower_tril_mat
                        )
                        assert (
                            torch.norm(cov - approx_cov) < 1e-7
                        ), f"Diff: {torch.norm(cov - approx_cov)}"
                        # print(f"Lower tril cov: {lower_tril_cov[None, ...].shape}")
                        # print(f"Mu: {mu[None, ...].shape}")
                        # print(f"GT Choir: {gt_choir.shape}")
                        aug_gt_choir = torch.zeros(
                            gt_choir.shape[0], gt_choir.shape[1], 11
                        ).to(gt_choir.device)
                        # print(f"Anchor indices: {self._anchor_indices.shape}")
                        mu = mu.to(aug_gt_choir.device)
                        lower_tril_cov = lower_tril_cov.to(aug_gt_choir.device)
                        # TODO: Remove this. It's temporary (I know it can be vectorized but it
                        # wouldn't need to if it was done in the compute_choir function).
                        for i in range(self._anchor_indices.shape[0]):
                            # Concatenate the Gaussian parameters of anchor associated to the current
                            # index to the CHOIR field
                            anchor_idx = self._anchor_indices[i].item()
                            aug_gt_choir[:, i] = torch.cat(
                                [
                                    gt_choir[:, i],
                                    mu[None, anchor_idx],
                                    lower_tril_cov[None, anchor_idx],
                                ],
                                dim=-1,
                            )

                        # print(f"GT Choir w/ contacts: {aug_gt_choir.shape}")
                        # TODO: Make sure the distances in CHOIR are roughly the same scale as the
                        # Gaussian parameters. If not we'll have to adjust the exponential_map_w or
                        # rescale the Gaussian parameters.
                        # print(f"CHOIR range: {gt_choir.min()} - {gt_choir.max()}")
                        # print(
                        # f"Lower tril cov range: {lower_tril_cov.min()} - {lower_tril_cov.max()}"
                        # )
                        # print(
                        # f"Mu range: {mu.min()} - {mu.max()}"
                        # )
                    # =================================================================

                    sample_paths = []
                    # ================= Perturbed Hand-Object pairs =================
                    for j in range(self._seq_len):
                        sample_pth = osp.join(
                            samples_labels_pickle_pth, grasp_name, f"sample_{j:06d}.pkl"
                        )
                        if osp.isfile(sample_pth):
                            sample_paths.append(sample_pth)
                            continue

                        theta, beta, trans = (
                            gt_pose.clone(),
                            gt_shape.clone(),
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
                                torch.randn(45, device=theta.device)
                                * self._perturbations[self._perturbation_level][
                                    "axisangle"
                                ],
                            ]
                        )
                        theta += pose_noise
                        trans += trans_noise

                        verts, joints = affine_mano(theta, beta, trans)

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
                            obj_ptcld.unsqueeze(0),
                            anchors,
                            scalar=scalar,
                            bps=self._bps.unsqueeze(0),  # type: ignore
                            anchor_indices=self._anchor_indices,  # type: ignore
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
                            torch.zeros(6).cpu().numpy(),
                            trans.squeeze().cpu().numpy(),
                            joints.squeeze().cpu().numpy(),
                            anchors.squeeze().cpu().numpy(),
                        ), (
                            aug_gt_choir.squeeze().cpu().numpy()
                            if self._model_contacts
                            else gt_choir.squeeze().cpu().numpy(),
                            gt_rescaled_ref_pts.squeeze().cpu().numpy(),
                            gt_scalar.cpu().numpy(),
                            gt_joints.squeeze().cpu().numpy(),
                            gt_anchors.squeeze().cpu().numpy(),
                            gt_pose.squeeze().cpu().numpy(),
                            gt_shape.squeeze().cpu().numpy(),
                            torch.zeros(6).cpu().numpy(),
                            gt_trans.squeeze().cpu().numpy(),
                            gt_contact_gaussian_params.squeeze().cpu().numpy()
                            if self._model_contacts
                            else torch.zeros(1).cpu().numpy(),
                        )
                        # =================================================================

                        with open(sample_pth, "wb") as f:
                            pkl = pickle.dumps((sample, label, obj_pth))
                            compressed_pkl = blosc2.compress(pkl)
                            f.write(compressed_pkl)
                        sample_paths.append(sample_pth)

                        if visualize and not has_visualized:
                            print(
                                f"[*] Plotting CHOIR for {grasp_name} ... (please be patient)"
                            )
                            # visualize_CHOIR(
                            # gt_choir.squeeze(0),
                            # self._bps,
                            # self._anchor_indices,
                            # scalar,
                            # gt_verts.squeeze(0),
                            # gt_anchors.squeeze(0),
                            # obj_mesh,
                            # obj_ptcld,
                            # gt_rescaled_ref_pts.squeeze(0),
                            # affine_mano,
                            # use_deltas=self._use_deltas,
                            # )

                            # visualize_CHOIR(
                            # choir.squeeze(0),
                            # self._bps,
                            # self._anchor_indices,
                            # scalar,
                            # verts.squeeze(0),
                            # anchors.squeeze(0),
                            # obj_mesh,
                            # obj_ptcld,
                            # rescaled_ref_pts.squeeze(0),
                            # affine_mano,
                            # use_deltas=self._use_deltas,
                            # )
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
                            continue
                            # visualize_CHOIR_prediction(
                            # gt_choir,
                            # gt_choir,
                            # self._bps,
                            # self._anchor_indices,
                            # scalar,
                            # gt_scalar,
                            # rescaled_ref_pts,
                            # gt_rescaled_ref_pts,
                            # gt_verts,
                            # gt_joints,
                            # gt_anchors,
                            # is_rhand=(hand_idx == "right"),
                            # use_smplx=False,
                            # dataset="ContactPose",
                            # remap_bps_distances=self._remap_bps_distances,
                            # exponential_map_w=self._exponential_map_w,
                            # use_deltas=self._use_deltas,
                            # )
                            # ==============================
                            colours = np.zeros_like(gt_verts[0].detach().cpu().numpy())
                            print(
                                f"[*] Computed contacts: {gt_vertex_contacts.shape}. "
                                + f"Range: {gt_vertex_contacts.min()} - {gt_vertex_contacts.max()}"
                            )
                            print(f"[*] Colours: {colours.shape}")
                            # Visualize contacts by colouring the vertices
                            colours[:, 0] = (
                                gt_vertex_contacts / gt_vertex_contacts.max()
                            )
                            colours[:, 1] = 0.58 - colours[:, 0]
                            colours[:, 2] = 0.66 - colours[:, 0]
                            gt_hand_mesh.vertex_colors = o3du.Vector3dVector(colours)
                            open3d.visualization.draw_geometries(
                                [gt_hand_mesh, obj_mesh]
                            )
                            # o3dv.draw_geometries([hand_mesh])
                            for i in range(32):
                                break
                                (
                                    gaussian_params,
                                    anchor_contacts,
                                ) = compute_anchor_gaussians(
                                    gt_verts[0],
                                    gt_anchors[0],
                                    gt_vertex_contacts,
                                    base_unit=self.base_unit,
                                    anchor_mean_threshold_mm=20,
                                    min_contact_points_for_neighbourhood=5,
                                    debug_anchor=i,
                                )
                                print(f"[*] Gaussian params: {gaussian_params.shape}")
                                # Visualize the Gaussian parameters
                                visualize_3D_gaussians_on_hand_mesh(
                                    gt_hand_mesh,
                                    obj_mesh,
                                    gaussian_params,
                                    base_unit=self.base_unit,
                                    debug_anchor=i,
                                    anchors=gt_anchors[0],
                                    anchor_contacts=anchor_contacts,  # Only for debugging
                                )
                            print(
                                f"[*] Gaussian params: {gt_contact_gaussian_params.shape}"
                            )
                            # Visualize the Gaussian parameters
                            visualize_3D_gaussians_on_hand_mesh(
                                gt_hand_mesh,
                                obj_mesh,
                                gt_contact_gaussian_params,
                                base_unit=self.base_unit,
                                anchors=gt_anchors[0],
                            )

                            print(
                                "====== Reconstructing contacts from 3D Gaussians ======"
                            )
                            visualize_hand_contacts_from_3D_gaussians(
                                gt_hand_mesh,
                                gt_contact_gaussian_params,
                                gt_anchors[0],
                                gt_contacts=gt_vertex_contacts,
                            )
                            print(
                                f"gaussian params shape: {gt_contact_gaussian_params.shape}"
                            )
                            mu = gt_contact_gaussian_params[:, :3]
                            cov = gt_contact_gaussian_params[:, 3:].reshape(-1, 3, 3)
                            cholesky_cov = torch.zeros_like(cov)
                            for i in range(cov.shape[0]):
                                if torch.all(cov[i] == 0):
                                    continue
                                cholesky_cov[i] = torch.linalg.cholesky(cov[i])
                                print(
                                    f"Cholesky shape: {cholesky_cov[i].shape}. Random element: {cholesky_cov[i]}"
                                )
                            approx_cov = torch.einsum(
                                "bik,bjk->bij", cholesky_cov, cholesky_cov
                            )
                            print(f"Diff: {torch.norm(cov - approx_cov)}")
                            choleksy_gaussian_params = torch.cat(
                                (mu, approx_cov.reshape(-1, 9)), dim=-1
                            )
                            # Visualize the Gaussian parameters
                            visualize_3D_gaussians_on_hand_mesh(
                                gt_hand_mesh,
                                obj_mesh,
                                choleksy_gaussian_params,
                                base_unit=self.base_unit,
                                anchors=gt_anchors[0],
                            )

                            print(
                                "====== Reconstructing contacts from 3D Gaussians ======"
                            )
                            visualize_hand_contacts_from_3D_gaussians(
                                gt_hand_mesh,
                                choleksy_gaussian_params,
                                gt_anchors[0],
                                gt_contacts=gt_vertex_contacts,
                            )
                            # visualize_CHOIR_prediction(
                            # choir,
                            # gt_choir,
                            # self._bps,
                            # self._anchor_indices,
                            # gt_scalar,
                            # gt_scalar,
                            # gt_rescaled_ref_pts,
                            # gt_rescaled_ref_pts,
                            # gt_verts,
                            # gt_joints,
                            # gt_anchors,
                            # is_rhand=(hand_idx == "right"),
                            # contact_gaussians=None,
                            # use_smplx=False,
                            # dataset="ContactPose",
                            # remap_bps_distances=self._remap_bps_distances,
                            # exponential_map_w=self._exponential_map_w,
                            # use_deltas=self._use_deltas,
                            # plot_choir=False,
                            # )
                            # # Now augment the noisy CHOIR with the Gaussian parameters
                            obj_mesh.compute_vertex_normals()
                            obj_vert_normals = torch.cat(
                                (
                                    torch.from_numpy(
                                        np.asarray(obj_mesh.vertices)
                                    ).type(dtype=torch.float32),
                                    torch.from_numpy(
                                        np.asarray(obj_mesh.vertex_normals)
                                    ).type(dtype=torch.float32),
                                ),
                                dim=-1,
                            )
                            # visualize_CHOIR_prediction(
                            # choir,
                            # gt_choir,
                            # self._bps,
                            # self._anchor_indices,
                            # gt_scalar,
                            # gt_scalar,
                            # gt_rescaled_ref_pts,
                            # gt_rescaled_ref_pts,
                            # gt_verts,
                            # gt_joints,
                            # gt_anchors,
                            # contact_gaussians=gt_contact_gaussian_params,
                            # obj_pts=obj_ptcld.float(),
                            # obj_normals=obj_vert_normals,
                            # is_rhand=(hand_idx == "right"),
                            # use_smplx=False,
                            # dataset="ContactPose",
                            # remap_bps_distances=self._remap_bps_distances,
                            # exponential_map_w=self._exponential_map_w,
                            # use_deltas=self._use_deltas,
                            # plot_choir=False,
                            # )

                            # visualize_CHOIR_prediction(
                            # choir,
                            # gt_choir,
                            # self._bps,
                            # self._anchor_indices,
                            # gt_scalar,
                            # gt_scalar,
                            # gt_rescaled_ref_pts,
                            # gt_rescaled_ref_pts,
                            # gt_verts,
                            # gt_joints,
                            # gt_anchors,
                            # contact_gaussians=choleksy_gaussian_params,
                            # obj_pts=obj_ptcld.float(),
                            # obj_normals=obj_vert_normals,
                            # is_rhand=(hand_idx == "right"),
                            # use_smplx=False,
                            # dataset="ContactPose",
                            # remap_bps_distances=self._remap_bps_distances,
                            # exponential_map_w=self._exponential_map_w,
                            # use_deltas=self._use_deltas,
                            # plot_choir=False,
                            # )
                            # ==============================
                            # ==============================
                            has_visualized = True
                    grasp_paths.append(sample_paths)
                pbar.update()
        if computed:
            print(
                f"[*] Dataset MPJPE (mm): {dataset_mpjpe.compute().item() * self.base_unit}"
            )
            print(
                f"[*] Dataset Root-aligned MPJPE (mm): {dataset_root_aligned_mpjpe.compute().item() * self.base_unit}"
            )
        return grasp_paths
