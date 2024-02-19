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
from typing import List, Tuple

import numpy as np
import torch
from hydra.utils import get_original_cwd
from metabatch import TaskSet
from oikit.oi_shape import OakInkShape
from tqdm import tqdm
from trimesh import Trimesh

from utils import debug_methods
from utils.dataset import augment_hand_object_pose, sample_udf
from utils.helpers import compressed_read, compressed_write
from utils.viz import make_mesh, plot_udf


@debug_methods
class OakInkDataset(TaskSet):
    def __init__(
        self,
        dataset_root: str,
        min_ctx_pts: int,
        max_ctx_pts: int,
        max_tgt_pts: int,
        total_tgt_pts: int,
        points_per_udf: int,
        predict_full_target: bool,
        use_squared_dist: bool,
        augment: bool,
        normalize: bool,
        debug: bool,
        split: str,
        seed: int,
        dataset_name: str,
        tiny: bool = False,
    ) -> None:
        super().__init__(
            min_pts=min_ctx_pts,
            max_ctx_pts=max_ctx_pts,
            max_tgt_pts=max_tgt_pts,
            total_tgt_pts=total_tgt_pts,
            eval=split != "train",
            predict_full_target=predict_full_target,
            predict_full_target_during_eval=True,
        )
        self._points_per_udf = points_per_udf
        self._augment = augment and split == "train"
        self._normalize = normalize
        self._debug = debug
        self._dataset_name = dataset_name
        self._cache_dir = osp.join(
            get_original_cwd(), "data", f"{dataset_name}_preprocessed"
        )
        self._n_pts = total_tgt_pts
        self._sigmas = [0.1, 0.08, 0.02, 0.003]
        self._sampling_ratios = [0.1, 0.04, 0.38, 0.48]
        assert len(self._sigmas) == len(
            self._sampling_ratios
        ), "Must have same number of sigmas and sampling ratios."
        assert np.sum(self._sampling_ratios) == 1, "Sampling ratios must sum to 1."
        assert np.all(
            np.array(self._sampling_ratios) >= 0
        ), "Sampling ratios must be positive."
        self._use_squared_dist = use_squared_dist
        self._samples, self._labels = self._load(dataset_root, tiny, split, seed)

    def _load(
        self,
        dataset_root: str,
        tiny: bool,
        split: str,
        seed: int,
        # ) -> Tuple[Union[dict, list], Union[dict, list]]:
    ) -> List[str]:
        # 1. Check if it's on disk, then load.
        # 2. If not, generate and save.
        # 2.1. Go through all OakInk shapes.
        os.environ["OAKINK_DIR"] = dataset_root
        dataset = OakInkShape(
            data_split=split,
            mano_assets_root=os.path.join(dataset_root, "assets/mano_v1_2"),
        )
        hand_obj_pairs = []
        idx = 0
        samples_dir = osp.join(
            self._cache_dir,
            f"{split}_samples_{self._n_pts}-pts_"
            + f"sigmas_{'-'.join([str(s) for s in self._sigmas])}_"
            + f"sampling_ratios_{'-'.join([str(s) for s in self._sampling_ratios])}"
            + f"_use_squared_dist_{self._use_squared_dist}"
            + "{'_tiny' if tiny else ''}"
            + f"_seed_{seed}",
        )
        print(f"[*] Generating UDFs for {'(tiny) ' if tiny else ''}split '{split}'...")
        pbar = tqdm(
            total=len(dataset) if not tiny else (1000 if split == "train" else 100)
        )
        for shape in dataset:
            if idx == (1000 if split == "train" else 50) and tiny:
                break
            #    2.2. For each shape, apply random rotation around the object's center and compute the
            #    hand pose in the object's coordinate frame.
            """
            Annotations: "Object's .obj models in its canonical system; MANO's pose & shape parameters
            and vertex 3D locations in object's canonical system."

            shape has the following structure:
            {
                seq_id: str
                obj_id: str
                joints: 21x3
                verts: 778x3
                hand_pose: axis-angle for MANO, (48,)
                hand_shape: beta for MANO, (10,)
                hand_tsl: translation for MANO, (3,). This is actually the translation for CENTER
                            JOINT, not necessarily the wrist. It's actually joint 9.
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
            pair_path = osp.join(samples_dir, shape["seq_id"])
            os.makedirs(pair_path, exist_ok=True)
            files = [
                osp.join(pair_path, f)
                for f in os.listdir(pair_path)
                if f.endswith(".pkl")
            ]
            if len(files) == len(self._sigmas):
                hand_obj_pairs.append(osp.dirname(files[0]))
                idx += 1
                pbar.update()
                continue

            obj_mesh = make_mesh(
                shape["obj_verts"], shape["obj_faces"], color=[1.0, 1.0, 0.0]
            )
            hand_mesh = make_mesh(
                shape["verts"],
                np.asarray(dataset.mano_layer.th_faces),
                color=[0.4, 0.81960784, 0.95294118],
            )

            if self._augment:
                augment_hand_object_pose(obj_mesh, hand_mesh, around_z=False)

            object_tmesh = Trimesh(
                vertices=shape["obj_verts"], faces=shape["obj_faces"]
            )
            hand_tmesh = Trimesh(
                vertices=shape["verts"], faces=dataset.mano_layer.th_faces
            )

            """
            The idea (from the UDF paper) is to sample points with different values of sigma to
            help with the learning of the UDF. They use 3 values: [0.08, 0.02, 0.003]. During
            training, they use sampling ratios such as [0.01, 0.49, 0.5] for each sigma value to
            determine how many points to sample with each sigma value such that they account for
            a certain percentage of the total number of points. They use 200k points in total and
            sample 100K points for each sigma value.

            Ok so after looking at their code, here is what I understand:
            1. They generate a UDF of 100K points for each sigma value and save it on disk.
            2. For each dataset sample, they concatenate the 3 UDFs as random subsets of
            N=(ratio*total_points) points in order to obtain total_points.
            3. The result is grid_coords, points and udf. But why do we need grid_coords (in range
            -1, 1)?? Points is in the range (-0.5, 0.5) and in mesh coordinate system. The two are
            the same points but in different scales. See this issue: https://github.com/jchibane/ndf/issues/18

            I may want to adapt this strategy a bit for my case, since the hand isn't centered on
            the origin as the object is. I could sample points around the object with different
            sigma values, then around the hand the same way, and then combine the two sets of
            points. Finally I could sample a much larger ball around the origin, such that it
            encodes the coarse UDF for the hand-object pair. This way we have several levels of
            granularity for the learning. For the observation set, I would need to try several
            stragies: (1) use the coarse set only, (2) use the joint dense sets (hand + object)
            only, (3) use the joint dense sets + the coarse set.
            """

            #    2.3. Generate a UDF for the object.
            #    2.4. Generate a UDF for the hand, as well as a coarse UDF using the
            #    MANO anchors.
            for sigma in self._sigmas:
                pair_udf, pair_grid_coords, pair_points = sample_udf(
                    object_tmesh,
                    hand_tmesh,
                    self._points_per_udf,
                    sigma=sigma,
                    use_squared_dist=self._use_squared_dist,
                    time_it=self._debug,
                )
                # Merge the two sets of points but the output UDF is 2D (object and hand DFs):
                # Save the UDFs:
                compressed_write(
                    osp.join(pair_path, f"udf_{sigma}.pkl"),
                    {
                        "udf": pair_udf,
                        "grid_coords": pair_grid_coords,
                        "points": pair_points,
                    },
                )
                if self._debug:
                    # Plot the object mesh and UDF as sampled points coloured by UDF value:
                    # plot_udf(obj_points, obj_udf, obj_mesh)
                    # Visualize the hand mesh and UDF as sampled points coloured by UDF value:
                    # plot_udf(hand_points, hand_udf, hand_mesh)
                    # Visualize the pair mesh and UDF as sampled points coloured by UDF value:
                    plot_udf(pair_points, pair_udf, meshes=[hand_mesh, obj_mesh])
            hand_obj_pairs.append(pair_path)
            idx += 1
            pbar.update()
        print(f"[*] Loaded {len(hand_obj_pairs)} hand-object pairs.")
        return hand_obj_pairs, None  # samples are HOI UDFs, labels are not needed.

    def __len__(self):
        return len(self._samples)

    def __gettask__(
        self, idx: int, n_context: int, n_target: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        def compute_udf_aggregate(
            n_pts: int,
            sampling_ratios: List[float],
            udfs_path: str,
        ):
            udf, grid_coords, points = [], [], []
            assert len([f for f in os.listdir(udfs_path) if f.endswith(".pkl")]) == len(
                sampling_ratios
            ), "Must have as many UDF pickles as sigma values."
            for ratio, sigma in zip(sampling_ratios, self._sigmas):
                sub_udf_dict = compressed_read(osp.join(udfs_path, f"udf_{sigma}.pkl"))
                subsample_indices = np.random.randint(
                    0, len(sub_udf_dict["udf"]), int(ratio * n_pts)
                )
                udf.extend(sub_udf_dict["udf"][subsample_indices])
                points.extend(sub_udf_dict["points"][subsample_indices])
                grid_coords.extend(sub_udf_dict["grid_coords"][subsample_indices])
            return {
                "df": np.array(udf, dtype=np.float32),
                "points": np.array(points, dtype=np.float32),
                "grid_coords": np.array(grid_coords, dtype=np.float32),
            }

        # 1. Load a sample from the dataset (random aggregate UDF).
        # print(f"Aggregating sub-UDFs for a total of {n_target} target points...")
        hoi_udf = compute_udf_aggregate(
            n_pts=n_target,
            sampling_ratios=self._sampling_ratios,
            udfs_path=self._samples[idx],
        )
        # print(f"Sampling {n_context} context points...")
        # 2. Sample a sparse point cloud from the UDF (D_train) as a subset of the UDF points.
        context_indices = np.random.randint(0, len(hoi_udf["df"]), n_context)
        context_udf = hoi_udf["df"][context_indices]
        context_points = hoi_udf["points"][context_indices]
        context_grid_coords = hoi_udf["grid_coords"][context_indices]
        # 3. The rest of the UDF is the dense point cloud (D_test).
        target_udf = hoi_udf["df"]
        target_points = hoi_udf["points"]
        target_grid_coords = hoi_udf["grid_coords"]
        # 4. Return (D_train, D_test).
        return (
            {
                "udf": torch.from_numpy(context_udf),
                "points": torch.from_numpy(context_points),
                "grid_coords": torch.from_numpy(context_grid_coords),
            },
            {
                "udf": torch.from_numpy(target_udf),
                "points": torch.from_numpy(target_points),
                "grid_coords": torch.from_numpy(target_grid_coords),
            },
        )
