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
from typing import List, Tuple

import blosc
import torch
from oikit.oi_shape import OakInkShape
from tqdm import tqdm
from trimesh import Trimesh

from dataset.base import BaseDataset
from utils import debug_methods


@debug_methods
class OakInkDataset(BaseDataset):
    base_unit = 1000.0  # The dataset is in meters, we want to work in mm. # TODO: Is it in meters???

    # ====== UDFs ======
    gt_udf_mean = torch.tensor([0.00, 0.00])  # object distances, hand distances
    gt_udf_std = torch.tensor([0.00, 0.00])  # object distances, hand distances
    noisy_udf_mean = torch.tensor([0.00, 0.00])  # object distances, hand distances
    noisy_udf_std = torch.tensor([0.00, 0.00])  # object distances, hand distances

    # ====== Keypoints ======
    gt_kp_obj_mean, gt_kp_hand_mean = torch.tensor([0, 0, 0]), torch.tensor([0, 0, 0])
    gt_kp_obj_std, gt_kp_hand_std = torch.tensor([0, 0, 0]), torch.tensor([0, 0, 0])
    noisy_kp_obj_mean, noisy_kp_hand_mean = torch.tensor([0, 0, 0]), torch.tensor(
        [0, 0, 0]
    )
    noisy_kp_obj_std, noisy_kp_hand_std = torch.tensor([0, 0, 0]), torch.tensor(
        [0, 0, 0]
    )

    # ====== Contacts ======
    contacts_mean = torch.tensor(
        [
            0,  # mean x
            0,  # mean y
            0,  # mean z
            0,  # cholesky-decomped cov 00
            0,  # cholesky-decomped cov 03
            0,  # cholesky-decomped cov 04
            0,  # cholesky-decomped cov 06
            0,  # cholesky-decomped cov 07
            0,  # cholesky-decomped cov 08
        ]
    )
    contacts_std = torch.tensor(
        [
            0.0,  # mean x
            0.0,  # mean y
            0.0,  # mean z
            0.0,  # cholesky-decomped cov 00
            0.0,  # cholesky-decomped cov 03
            0.0,  # cholesky-decomped cov 04
            0.0,  # cholesky-decomped cov 06
            0.0,  # cholesky-decomped cov 07
            0.0,  # cholesky-decomped cov 08
        ]
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

    def _load_objects_and_grasps(
        self, tiny: bool, split: str, seed: int = 0
    ) -> Tuple[dict, list, list, str]:
        os.environ["OAKINK_DIR"] = self._dataset_root
        dataset = OakInkShape(
            data_split=split,
            mano_assets_root="vendor/manotorch/assets/",
        )
        idx = 0
        n_samples = len(dataset) if not tiny else (1000 if split == "train" else 100)
        dataset_path = osp.join(
            self._cache_dir,
            f"{split}_{n_samples}-samples"
            + f"_{self._obj_ptcld_size}-obj-pts"
            + f"{'_tiny' if tiny else ''}"
            + f"{'right-hand' if self._right_hand_only else 'both-hands'}_seed-{seed}.pkl",
        )
        objects, grasps = [], []
        print(f"[*] Loading OakInk{' (tiny)' if tiny else ''}...")
        if osp.isfile(dataset_path):
            with open(dataset_path, "rb") as f:
                compressed_pkl = f.read()
                objects, grasps = pickle.loads(blosc.decompress(compressed_pkl))
        else:
            pbar = tqdm(total=n_samples)
            for shape in dataset:
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
                    hand_pose: axis-angle pose for MANO, (16x4,) in quaternion in obj-canonical space
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
                print(list(shape.keys()))
                obj_path = osp.join(dataset_path, f"{shape['obj_id']}.obj")
                obj_tmesh = Trimesh(
                    vertices=shape["obj_verts"], faces=shape["obj_faces"]
                )
                with open(obj_path, "w") as f:
                    f.write(obj_tmesh.export("obj"))
                grasp_path = osp.join(
                    dataset_path, f"{shape['obj_id']}_grasp_{idx}.pkl"
                )
                with open(grasp_path, "wb") as f:
                    pickle.dump(
                        {
                            "joints": shape["joints"],
                            "verts": shape["verts"],
                            "hand_pose": shape["hand_pose"],
                            "hand_shape": shape["hand_shape"],
                            "hand_tsl": shape["hand_tsl"],
                        },
                        f,
                    )

                objects.append(obj_path)
                grasps.append(grasp_path)
                idx += 1
                pbar.update()
            with open(dataset_path, "wb") as f:
                pkl = pickle.dumps((objects, grasps))
                compressed_pkl = blosc.compress(pkl)
                f.write(compressed_pkl)
        print(f"[*] Loaded {len(objects)} objects and {len(grasps)} grasps.")
        assert len(objects) == len(grasps)
        return (
            objects,
            grasps,
            osp.basename(dataset_path.split(".")[0]),
        )

    def _load(
        self,
        dataset_root: str,
        tiny: bool,
        split: str,
        seed: int,
        # ) -> Tuple[Union[dict, list], Union[dict, list]]:
    ) -> List[str]:
        raise NotImplementedError("This method is not implemented.")

    def __len__(self):
        return len(self._samples)
