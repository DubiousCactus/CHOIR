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

import blosc2
import numpy as np
import torch
from hydra.utils import get_original_cwd
from open3d import geometry as o3dg
from open3d import io as o3dio
from open3d import utility as o3du
from open3d import visualization as o3dv
from pytorch3d.transforms import matrix_to_rotation_6d
from torchmetrics import MeanMetric
from tqdm import tqdm
from trimesh import Trimesh

from conf.project import ANSI_COLORS, Theme
from dataset.base import BaseDataset
from globals import FLAT_LOWER_INDICES
from model.affine_mano import AffineMANO
from utils import colorize
from utils.dataset import (
    augment_hand_object_pose,
    compute_anchor_gaussians,
    compute_choir,
    get_contact_counts_by_neighbourhoods,
    get_scalar,
)
from utils.visualization import (
    visualize_3D_gaussians_on_hand_mesh,
    visualize_CHOIR,
    visualize_hand_contacts_from_3D_gaussians,
    visualize_MANO,
)


class ContactPoseDataset(BaseDataset):
    """ "
    A task is defined as a set of random views of the same grasp of the same object. In ContactPose
    terms, this translates to a set of random views of person X grasping object Y with hand H and
    intent Z.
    """

    base_unit = 1000.0  # The dataset is in meters, we want to work in mm.

    # ====== UDFs ======
    gt_udf_mean = torch.tensor([0.7469, 0.5524])  # object distances, hand distances
    gt_udf_std = torch.tensor([0.1128, 0.1270])  # object distances, hand distances
    noisy_udf_mean = torch.tensor([0.7469, 0.4955])  # object distances, hand distances
    noisy_udf_std = torch.tensor([0.1128, 0.1432])  # object distances, hand distances

    # ====== Keypoints ======
    gt_kp_obj_mean, gt_kp_hand_mean = torch.tensor(
        [1.3319e-04, 3.8958e-05, 2.8398e-05]
    ), torch.tensor([3.4145e-04, -4.4793e-04, 4.5160e-05])
    gt_kp_obj_std, gt_kp_hand_std = torch.tensor(
        [0.0404, 0.0390, 0.0389]
    ), torch.tensor([0.0485, 0.0456, 0.0448])
    noisy_kp_obj_mean, noisy_kp_hand_mean = torch.tensor(
        [-2.2395e-04, -8.6842e-04, 7.1034e-05]
    ), torch.tensor([0.0003, -0.0003, 0.0007])
    noisy_kp_obj_std, noisy_kp_hand_std = torch.tensor(
        [0.0383, 0.0369, 0.0368]
    ), torch.tensor([0.0712, 0.0696, 0.0691])

    # ====== Contacts ======
    contacts_mean = torch.tensor(
        [
            -4.8649e-07,  # mean x
            1.0232e-06,  # mean y
            7.0423e-06,  # mean z
            1.0232e-03,  # cholesky-decomped cov 00
            -1.1730e-06,  # cholesky-decomped cov 03
            8.0673e-04,  # cholesky-decomped cov 04
            -3.6392e-07,  # cholesky-decomped cov 06
            1.6037e-06,  # cholesky-decomped cov 07
            5.6080e-04,  # cholesky-decomped cov 08
        ]
    )
    contacts_std = torch.tensor(
        [
            0.0017,  # mean x
            0.0017,  # mean y
            0.0017,  # mean z
            0.0017,  # cholesky-decomped cov 00 *
            0.0011,  # cholesky-decomped cov 03
            0.0014,  # cholesky-decomped cov 04 *
            0.0011,  # cholesky-decomped cov 06
            0.0011,  # cholesky-decomped cov 07
            0.0011,  # cholesky-decomped cov 08 *
        ]
    )

    contacts_min = torch.tensor(
        [-0.0195, -0.0197, -0.0196, 0.0000, -0.0119, 0.0000, -0.0113, -0.0113, 0.0000]
    )
    contacts_max = torch.tensor(
        [0.0195, 0.0195, 0.0194, 0.0129, 0.0108, 0.0129, 0.0112, 0.0105, 0.0117]
    )

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
            {"trans": 0.0, "rot": 0.0, "pca": 0.0},  # Level 0
            {"trans": 0.02, "rot": 0.05, "pca": 0.3},  # Level 1
            {
                "trans": 0.05,
                "rot": 0.15,
                "pca": 0.5,
            },  # Level 2 (0.05m, 0.15rad, 0.5 PCA units)
            {
                "trans": 0.005,
                "rot": 0.05,
                "pca": 0.5,
            },  # Level 3 (0.005m, 0.15rad, 0.5 PCA units), for tests with TTO
        ]
        self._use_contactopt_splits = use_contactopt_splits
        self._use_improved_contactopt_splits = use_improved_contactopt_splits
        # Using ContactOpt's parameters
        if self._use_contactopt_splits or self._use_improved_contactopt_splits:
            if split == "train":
                noisy_samples_per_grasp = 16
            else:
                noisy_samples_per_grasp = 4
        if split != "train":
            noisy_samples_per_grasp = 4  # We also want that for object-based splits

        self._eval_observation_plateau = eval_observations_plateau
        if eval_observations_plateau:
            if split == "test":
                noisy_samples_per_grasp = 15

        if eval_anchor_assignment:
            noisy_samples_per_grasp = 1
            max_views_per_grasp = 1

        self._eval_anchor_assignment = eval_anchor_assignment
        self._model_contacts = model_contacts

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
        np.int = int  # If using python 3.11
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
        n_participants = 5 if tiny else 51
        for p_num in range(1, n_participants):
            for intent in ["use", "handoff"]:
                for obj_name in get_object_names(
                    p_num, intent
                ):  # TODO Fix this non-deterministic pieace of shit function!!!!!
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
                    blosc2.decompress(compressed_pkl)
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
                compressed_pkl = blosc2.compress(pkl)
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
        # a bit hacky and not scalable. But for readability and debugging it's better to
        # automatically generate the cache key from the class properties in human-readable form.
        samples_labels_pickle_pth = osp.join(
            self._cache_dir,
            "samples_and_labels",
            # f"dataset_{hashlib.shake_256(dataset_name.encode()).hexdigest(8)}_"
            f"perturbed-{self._perturbation_level}_"
            + f"_{self._obj_ptcld_size}-obj-pts"
            + f"_{'right-hand' if self._right_hand_only else 'both-hands'}"
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
        affine_mano: AffineMANO = AffineMANO(for_contactpose=True)  # type: ignore

        n_augs = self._n_augs if self._augment else 0

        # For each object-grasp pair, compute the CHOIR field.
        print("[*] Computing CHOIR fields...")
        pbar = tqdm(total=len(objects_w_contacts) * (n_augs + 1))
        dataset_mpjpe = MeanMetric()
        dataset_root_aligned_mpjpe = MeanMetric()
        computed = False
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
                    visualize = self._debug and (random.Random().random() < 0.8)
                    has_visualized = False
                    computed = True
                    # ================== Original Hand-Object Pair ==================
                    mano_params = grasp_data["grasp"][0].copy()
                    gt_hTm = torch.from_numpy(mano_params["hTm"]).float().unsqueeze(0)
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
                        torch.tensor(mano_params["pose"]).unsqueeze(0),
                        torch.tensor(mano_params["betas"]).unsqueeze(0),
                    )
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
                    # ================ Compute GT anchors and verts ==================
                    gt_verts, gt_joints = affine_mano(
                        gt_theta, gt_beta, gt_trans, rot_6d=gt_rot_6d
                    )
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
                            anchor_mean_threshold_mm=20,
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

                        verts, joints = affine_mano(theta, beta, trans, rot_6d=rot_6d)

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
                            rot_6d.squeeze().cpu().numpy(),
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
                            gt_theta.squeeze().cpu().numpy(),
                            gt_beta.squeeze().cpu().numpy(),
                            gt_rot_6d.squeeze().cpu().numpy(),
                            gt_trans.squeeze().cpu().numpy(),
                            gt_contact_gaussian_params.squeeze().cpu().numpy()
                            if self._model_contacts
                            else torch.zeros(1).cpu().numpy(),
                        )
                        # =================================================================

                        with open(sample_pth, "wb") as f:
                            pkl = pickle.dumps((sample, label, mesh_pth))
                            compressed_pkl = blosc2.compress(pkl)
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
                            o3dv.draw_geometries([gt_hand_mesh, obj_mesh])
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
                            # obj_mesh.compute_vertex_normals()
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
