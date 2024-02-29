#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.


import json
import os
import pickle
import signal
from collections import defaultdict
from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import blosc2
import matplotlib.pyplot as plt
import open3d.io as o3dio
import pyvista as pv
import torch
import trimesh
from ema_pytorch import EMA
from hydra.core.hydra_config import HydraConfig
from pytorch3d.transforms.rotation_conversions import rotation_6d_to_matrix
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric
from tqdm import tqdm
from trimesh import Trimesh

from conf import project as project_conf
from model.affine_mano import AffineMANO
from src.multiview_trainer import MultiViewTrainer
from utils import colorize, to_cuda, to_cuda_
from utils.dataset import lower_tril_cholesky_to_covmat
from utils.testing import (
    compute_binary_contacts,
    compute_contact_coverage,
    compute_contact_fidelity,
    compute_mpjpe,
    make_batch_of_obj_data,
    mp_compute_solid_intersection_volume,
    mp_process_obj_meshes,
)
from utils.training import (
    optimize_mesh_from_joints_and_anchors,
    optimize_pose_pca_from_choir,
)
from utils.visualization import (
    ScenePicAnim,
    visualize_3D_gaussians_on_hand_mesh,
    visualize_hand_contacts_from_3D_gaussians,
    visualize_MANO,
    visualize_model_predictions_with_multiple_views,
)


class MultiViewTester(MultiViewTrainer):
    def __init__(
        self,
        run_name: str,
        data_loader: DataLoader,
        model: torch.nn.Module,
        model_ckpt_path: str,
        training_loss: torch.nn.Module,
        **kwargs,
    ) -> None:
        """Base trainer class.
        Args:
            model (torch.nn.Module): Model to train.
            opt (torch.optim.Optimizer): Optimizer to use.
            data_loader (torch.utils.data.DataLoader): Training dataloader.
            val_loader (torch.utils.data.DataLoader): Validation dataloader.
        """
        self._is_baseline = False
        self._run_name = run_name
        self._model = model
        self._ema = EMA(
            self._model, beta=0.9999, update_after_step=100, update_every=10
        )
        assert "max_observations" in kwargs, "max_observations must be provided."
        assert "save_predictions" in kwargs, "save_predictions must be provided."
        self._max_observations = kwargs["max_observations"]
        self._save_predictions = kwargs.get("save_predictions", False)
        if model_ckpt_path is None:
            print(
                colorize(
                    "[!] No checkpoint path provided.", project_conf.ANSI_COLORS["red"]
                )
            )
        else:
            self._load_checkpoint(model_ckpt_path, model_only=True)
        self._training_loss = training_loss
        self._data_loader = data_loader
        self._running = True
        self._pbar = None
        self._affine_mano = to_cuda_(
            AffineMANO(
                ncomps=self._data_loader.dataset.theta_dim - 3,
                flat_hand_mean=(self._data_loader.dataset.name == "grab"),
                for_contactpose=(self._data_loader.dataset.name == "contactpose"),
                for_oakink=(self._data_loader.dataset.name == "oakink"),
            )
        )
        self._bps_dim = data_loader.dataset.bps_dim
        self._bps = to_cuda_(data_loader.dataset.bps)
        self._anchor_indices = to_cuda_(data_loader.dataset.anchor_indices)
        self._remap_bps_distances = data_loader.dataset.remap_bps_distances
        self._exponential_map_w = data_loader.dataset.exponential_map_w
        self._n_ctrl_c = 0
        self._enable_contacts_tto = kwargs.get("enable_contacts_tto", False)
        self._compute_iv = kwargs.get("compute_iv", False)
        self._compute_contact_fidelity = kwargs.get("compute_contact_fidelity", False)
        self._object_cache = {}
        self._pitch_mm = 2
        self._pitch = self._pitch_mm / self._data_loader.dataset.base_unit
        self._radius = int(
            0.2 / self._pitch
        )  # 20cm in each direction for the voxel grid
        self._n_pts_on_mesh = 5000
        self._n_normals_on_mesh = 5000
        self._data_loader.dataset.set_eval_mode(True)
        signal.signal(signal.SIGINT, self._terminator)

    @to_cuda
    def _visualize(
        self,
        batch: Union[Tuple, List, torch.Tensor],
        epoch: int,
    ) -> None:
        """Visualize the model predictions.
        Args:multiviewtes
            batch: The batch to process.
            epoch: The current epoch.
        """
        visualize_model_predictions_with_multiple_views(
            self._model,
            batch,
            epoch,
            bps_dim=self._bps_dim,
            bps=self._bps,
            anchor_indices=self._anchor_indices,
            remap_bps_distances=self._remap_bps_distances,
            exponential_map_w=self._exponential_map_w,
            dataset=self._data_loader.dataset.name,
            theta_dim=self._data_loader.dataset.theta_dim,
        )  # User implementation goes here (utils/visualization.py)

    def _inference(
        self,
        samples,
        labels,
        modality: str = "noisy_pair",
        max_observations: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns: {"choir": torch.Tensor, ?}
        """
        max_observations = max_observations or samples["choir"].shape[1]
        if kwargs["use_prior"]:
            y_hat = self._model(samples["choir"][:, :max_observations], use_mean=True)
        else:
            y_hat = self._model(
                samples["choir"][:, :max_observations], labels["choir"], use_mean=True
            )
        return y_hat

    def _compute_eval_metrics(
        self,
        anchors_pred: torch.tensor,
        verts_pred: torch.Tensor,
        joints_pred: torch.Tensor,
        gt_anchors: torch.Tensor,
        gt_verts: torch.Tensor,
        gt_joints: torch.Tensor,
        batch_obj_data: Dict,
    ):
        with torch.no_grad():
            # === Anchor error ===
            anchor_error = (
                torch.norm(anchors_pred - gt_anchors.to(anchors_pred.device), dim=2)
                .mean(dim=1)
                .mean(dim=0)
                .item()
                * self._data_loader.dataset.base_unit
            )
            # === MPJPE ===
            mpjpe, root_aligned_mpjpe = (
                x.item() for x in compute_mpjpe(gt_joints, joints_pred)
            )
            mpjpe *= self._data_loader.dataset.base_unit
            root_aligned_mpjpe *= self._data_loader.dataset.base_unit
            if self._data_loader.dataset.eval_observation_plateau:
                return (
                    anchor_error,
                    mpjpe,
                    root_aligned_mpjpe,
                    torch.zeros(1),
                    torch.zeros(1),
                    torch.zeros(1),
                )
            # ====== MPVPE ======
            # Compute the mean per-vertex position error (MPVPE) between the predicted and ground truth
            # hand meshes.
            pvpe = torch.linalg.vector_norm(
                gt_verts - verts_pred, ord=2, dim=-1
            )  # Per-vertex position error (B, 778, 3)
            mpvpe = torch.mean(pvpe, dim=-1)  # Mean per-vertex position error (B, 1)
            mpvpe = torch.mean(
                mpvpe, dim=0
            ).item()  # Mean per-vertex position error avgd across batch (1)
            mpvpe *= self._data_loader.dataset.base_unit
            # ====== Intersection volume ======
            # Let's now try by voxelizing the meshes and reporting the volume of voxels occupied by
            # both meshes:
            # TODO: WARNING!! For GRAB and approaching hands I can end up with empty hand voxels
            # with this radius!!! I overwrite it to 0.4 when evaluating on GRAB for now. Obviously
            # the thing to do is to skip empty voxels. I'll try to implement it.
            batch_intersection_volume = torch.inf
            if self._compute_iv:
                batch_intersection_volume = mp_compute_solid_intersection_volume(
                    batch_obj_data["voxel"],
                    hand_verts=verts_pred,
                    # Careful to use the closed faces as they should count for the hand volume!
                    mesh_faces=self._affine_mano.closed_faces.cpu().numpy(),
                    pitch=self._pitch,
                    radius=self._radius,
                )
            # ======= Contact Coverage =======
            # Percentage of hand points within 2mm of the object surface.
            batch_contact_coverage = compute_contact_coverage(
                gt_verts,
                # Careful not to use the closed faces as they shouldn't count for the hand surface points!
                self._affine_mano.faces,
                batch_obj_data["points"],
                thresh_mm=2,
                base_unit=self._data_loader.dataset.base_unit,
                n_samples=self._n_pts_on_mesh,
            )
            # ============ Contact fidelity ============
            batch_contact_fidelity = -torch.inf
            if self._compute_contact_fidelity:
                compute_bin_contacts = partial(
                    compute_binary_contacts,
                    faces=self._affine_mano.faces,
                    obj_points=batch_obj_data["points"],
                    thresh_mm=2,
                    base_unit=self._data_loader.dataset.base_unit,
                    n_mesh_upsamples=1,
                    return_upsampled_verts=False,
                )
                """
                n_mesh_upsamples=2 gives 12337 hand vertices instead of the original 778. Note that
                they aren't evenly distributed so it's not ideal, but as opposed to random sampling
                on the mesh, we can exactly recover contact points even on different poses.
                """
                pred_binary_contacts, _ = compute_bin_contacts(hand_verts=verts_pred)
                gt_binary_contacts, upsampled_verts = compute_bin_contacts(
                    hand_verts=gt_verts, return_upsampled_verts=True
                )
                batch_contact_fidelity = compute_contact_fidelity(
                    upsampled_verts, pred_binary_contacts, gt_binary_contacts
                )

        return {
            "Anchor error [mm] (↓)": anchor_error,
            "MPJPE [mm] (↓)": mpjpe,
            "Root-aligned MPJPE [mm] (↓)": root_aligned_mpjpe,
            "MPVPE [mm] (↓)": mpvpe,
            "Intersection volume [cm3] (↓)": batch_intersection_volume,
            "Contact coverage [%] (↑)": batch_contact_coverage,
            "Contact fidelity score (↑)": batch_contact_fidelity,
        }

    def _test_batch(
        self,
        samples: Dict,
        labels: Dict,
        mesh_pths: List[str],
        n_observations: int,
        batch_idx: int,
        use_prior: bool = True,
        use_input: bool = False,
    ) -> Tuple:
        input_scalar = samples["scalar"]
        if len(input_scalar.shape) == 2:
            input_scalar = input_scalar.mean(dim=1)
        cached_pred_path = f"cache_pred_{batch_idx}.pkl"
        if os.path.exists(cached_pred_path):
            with open(cached_pred_path, "rb") as f:
                y_hat = pickle.load(f)
                y_hat = to_cuda_(y_hat)
        else:
            y_hat = self._inference(samples, labels, use_prior=use_prior)
            with open(cached_pred_path, "wb") as f:
                y_hat = {k: v.detach().cpu() for k, v in y_hat.items()}
                pickle.dump(y_hat, f)

        mano_params_gt = {
            "pose": labels["theta"],
            "beta": labels["beta"],
            "rot_6d": labels["rot"],
            "trans": labels["trans"],
        }
        # Only use the last view for each batch element (they're all the same anyway for static
        # grasps, but for dynamic grasps we want to predict the LAST frame!).
        mano_params_gt = {k: v[:, -1] for k, v in mano_params_gt.items()}
        gt_pose, gt_shape, gt_rot_6d, gt_trans = tuple(mano_params_gt.values())
        gt_verts, gt_joints = self._affine_mano(
            gt_pose, gt_shape, gt_trans, rot_6d=gt_rot_6d
        )
        gt_anchors = self._affine_mano.get_anchors(gt_verts)
        if not self._data_loader.dataset.is_right_hand_only:
            raise NotImplementedError("Right hand only is implemented for testing.")
        multiple_obs = len(samples["theta"].shape) > 2
        gt_contact_gaussian_params = (
            labels["contact_gaussians"] if self._enable_contacts_tto else None
        )

        # ============== Object processing ==============
        # For mesh_pths we have a tuple of N lists of B entries. N is the number of
        # observations and B is the batch size. We'll take the last observation for each batch
        # element.
        mesh_pths = list(mesh_pths[-1])  # Now we have a list of B entries.
        for i in range(len(mesh_pths)):
            mesh_pths[i] = mesh_pths[i].replace(
                "/media/data3/moralest/", "/Users/cactus/Code/"
            )
            mesh_pths[i] = mesh_pths[i].replace("test_1000", "test_2000")
        print(f"Meshes: {mesh_pths}: len={len(mesh_pths)}. bs={gt_verts.shape[0]}")

        cached_obj_path = "batch_obj_data.pkl"
        if os.path.exists(cached_obj_path):
            with open(cached_obj_path, "rb") as f:
                # batch_obj_data = pickle.load(f, map_location="cpu")
                batch_obj_data = to_cuda_(
                    torch.load(f, map_location=torch.device("cpu"))
                )
        else:
            mp_process_obj_meshes(
                mesh_pths,
                self._object_cache,
                self._data_loader.dataset.center_on_object_com,
                self._enable_contacts_tto,
                self._compute_iv,
                self._pitch,
                self._radius,
                self._n_pts_on_mesh,
                self._n_normals_on_mesh,
                dataset=self._data_loader.dataset.name,
            )
            batch_obj_data = make_batch_of_obj_data(self._object_cache, mesh_pths)
        # ==============================================
        eval_metrics = {"Distance fitting only": None, "With contact fitting": None}
        if use_input:
            # For ground-truth:
            if not self._data_loader.dataset.eval_anchor_assignment:
                input_params = {
                    k: (v[:, -1] if multiple_obs else v)
                    for k, v in samples.items()
                    if k in ["theta", "beta", "rot", "trans"]
                }
                verts_pred, joints_pred = self._affine_mano(*input_params.values())
                anchors_pred = self._affine_mano.get_anchors(verts_pred)
                eval_metrics["input_udf"] = self._compute_eval_metrics(
                    anchors_pred,
                    verts_pred,
                    joints_pred,
                    gt_anchors,
                    gt_verts,
                    gt_joints,
                    batch_obj_data,
                )
            # For MANO fitting:
            else:
                with torch.set_grad_enabled(True):
                    (
                        _,
                        _,
                        _,
                        _,
                        anchors_pred,
                        verts_pred,
                        joints_pred,
                    ) = optimize_pose_pca_from_choir(
                        samples["choir"].squeeze(),
                        bps=self._bps,
                        anchor_indices=self._anchor_indices,
                        scalar=input_scalar,
                        max_iterations=4000,
                        loss_thresh=1e-10,
                        lr=1e-2,
                        is_rhand=samples["is_rhand"],
                        use_smplx=False,
                        dataset=self._data_loader.dataset.name,
                        remap_bps_distances=self._remap_bps_distances,
                        exponential_map_w=self._exponential_map_w,
                        initial_params=None,  # We want to see how well we can fit a randomly initialize MANO of course!
                        beta_w=1e-4,
                        theta_w=1e-7,
                        choir_w=1000,
                    )
                    eval_metrics["input_udf"] = self._compute_eval_metrics(
                        anchors_pred,
                        verts_pred,
                        joints_pred,
                        gt_anchors,
                        gt_verts,
                        gt_joints,
                        batch_obj_data,
                    )
        else:
            use_smplx = False  # TODO: I don't use it for now
            with torch.set_grad_enabled(True):
                if self._is_baseline:
                    joints_pred, anchors_pred = (
                        y_hat["hand_keypoints"][:, :21],
                        y_hat["hand_keypoints"][:, 21:],
                    )
                    verts_pred = optimize_mesh_from_joints_and_anchors(
                        y_hat["hand_keypoints"],
                        scalar=torch.mean(input_scalar)
                        .unsqueeze(0)
                        .to(input_scalar.device),  # TODO: What should I do here?
                        is_rhand=samples["is_rhand"][0],
                        max_iterations=400,
                        loss_thresh=1e-7,
                        lr=8e-2,
                        dataset=self._data_loader.dataset.name,
                        use_smplx=use_smplx,
                        initial_params={
                            k: (
                                v[:, -1] if multiple_obs else v
                            )  # Initial pose is the last observation
                            for k, v in samples.items()
                            if k
                            in [
                                "theta",
                                ("vtemp" if use_smplx else "beta"),
                                "rot",
                                "trans",
                            ]
                        },
                        beta_w=1e-4,
                        theta_w=1e-8,
                    )

                    eval_metrics["Distance fitting only"] = self._compute_eval_metrics(
                        anchors_pred,
                        verts_pred,
                        joints_pred,
                        gt_anchors,
                        gt_verts,
                        gt_joints,
                        batch_obj_data,
                    )
                else:
                    sample_to_viz = 2
                    contacts_pred, obj_points, obj_normals = (
                        None,
                        None,
                        None,
                    )
                    (
                        _,
                        _,
                        _,
                        _,
                        anchors_pred,
                        verts_pred,
                        joints_pred,
                    ) = optimize_pose_pca_from_choir(
                        y_hat["choir"],
                        contact_gaussians=contacts_pred
                        if self._enable_contacts_tto
                        else None,
                        obj_pts=obj_points,
                        obj_normals=obj_normals,
                        bps=self._bps,
                        anchor_indices=self._anchor_indices,
                        scalar=input_scalar,
                        max_iterations=1000,
                        loss_thresh=1e-7,
                        lr=8e-2,
                        is_rhand=samples["is_rhand"],
                        use_smplx=use_smplx,
                        dataset=self._data_loader.dataset.name,
                        remap_bps_distances=self._remap_bps_distances,
                        exponential_map_w=self._exponential_map_w,
                        initial_params={
                            k: (
                                v[:, -1] if multiple_obs else v
                            )  # Initial pose is the last observation
                            for k, v in samples.items()
                            if k
                            in [
                                "theta",
                                ("vtemp" if use_smplx else "beta"),
                                "rot",
                                "trans",
                            ]
                        },
                        beta_w=1e-4,
                        theta_w=1e-7,
                        choir_w=1000,
                    )
                    pred_hand_mesh = trimesh.Trimesh(
                        vertices=verts_pred[sample_to_viz].detach().cpu().numpy(),
                        faces=self._affine_mano.closed_faces.detach().cpu().numpy(),
                    )
                    gt_hand_mesh = trimesh.Trimesh(
                        vertices=gt_verts[sample_to_viz].detach().cpu().numpy(),
                        faces=self._affine_mano.closed_faces.detach().cpu().numpy(),
                    )
                    visualize_MANO(
                        pred_hand_mesh,
                        obj_mesh=batch_obj_data["mesh"][sample_to_viz],
                        gt_hand=gt_hand_mesh,
                    )
                    visualize_MANO(
                        pred_hand_mesh,
                        obj_mesh=batch_obj_data["mesh"][sample_to_viz],
                        opacity=1.0,
                    )
                    # eval_metrics["Distance fitting only"] = self._compute_eval_metrics(
                    # anchors_pred,
                    # verts_pred,
                    # joints_pred,
                    # gt_anchors,
                    # gt_verts,
                    # gt_joints,
                    # batch_obj_data,
                    # )
                    if self._enable_contacts_tto:
                        # del anchors_pred, verts_pred, joints_pred
                        contacts_pred, obj_points, obj_normals = (
                            y_hat.get("contacts", None),
                            batch_obj_data["points"],
                            batch_obj_data["normals"],
                        )
                        # Visualize the Gaussian parameters
                        print(
                            "====== Reconstructing contacts from GT 3D Gaussians ======"
                        )
                        print(
                            f"GT gaussian params shape: {gt_contact_gaussian_params[sample_to_viz, -1].shape}"
                        )
                        visualize_3D_gaussians_on_hand_mesh(
                            gt_hand_mesh,
                            batch_obj_data["mesh"][sample_to_viz],
                            gt_contact_gaussian_params[sample_to_viz, -1].cpu(),
                            base_unit=self._data_loader.dataset.base_unit,
                            anchors=gt_anchors[sample_to_viz].cpu(),
                        )
                        visualize_hand_contacts_from_3D_gaussians(
                            gt_hand_mesh,
                            gt_contact_gaussian_params[sample_to_viz, -1].cpu(),
                            gt_anchors[sample_to_viz].cpu(),
                        )

                        print("====== Reconstructing contacts from 3D Gaussians ======")
                        pred_contact_gaussian_params = torch.cat(
                            (
                                contacts_pred[sample_to_viz, ..., :3],
                                lower_tril_cholesky_to_covmat(contacts_pred[..., 3:])[
                                    sample_to_viz
                                ].view(-1, 9),
                            ),
                            dim=-1,
                        )
                        visualize_3D_gaussians_on_hand_mesh(
                            pred_hand_mesh,
                            batch_obj_data["mesh"][sample_to_viz],
                            pred_contact_gaussian_params.cpu(),
                            base_unit=self._data_loader.dataset.base_unit,
                            anchors=anchors_pred[sample_to_viz].cpu(),
                        )
                        visualize_hand_contacts_from_3D_gaussians(
                            pred_hand_mesh,
                            pred_contact_gaussian_params.cpu(),
                            anchors_pred[sample_to_viz].cpu(),
                        )

                        (
                            _,
                            _,
                            _,
                            _,
                            anchors_pred,
                            verts_pred,
                            joints_pred,
                        ) = optimize_pose_pca_from_choir(
                            y_hat["choir"],
                            contact_gaussians=contacts_pred
                            if self._enable_contacts_tto
                            else None,
                            obj_pts=obj_points,
                            obj_normals=obj_normals,
                            bps=self._bps,
                            anchor_indices=self._anchor_indices,
                            scalar=input_scalar,
                            max_iterations=1000,
                            loss_thresh=1e-7,
                            lr=8e-2,
                            is_rhand=samples["is_rhand"],
                            use_smplx=use_smplx,
                            dataset=self._data_loader.dataset.name,
                            remap_bps_distances=self._remap_bps_distances,
                            exponential_map_w=self._exponential_map_w,
                            initial_params={
                                k: (
                                    v[:, -1] if multiple_obs else v
                                )  # Initial pose is the last observation
                                for k, v in samples.items()
                                if k
                                in [
                                    "theta",
                                    ("vtemp" if use_smplx else "beta"),
                                    "rot",
                                    "trans",
                                ]
                            },
                            beta_w=1e-4,
                            theta_w=1e-7,
                            choir_w=1000,
                        )
                        pred_hand_mesh = trimesh.Trimesh(
                            vertices=verts_pred[-1].detach().cpu().numpy(),
                            faces=self._affine_mano.closed_faces.detach().cpu().numpy(),
                        )
                        gt_hand_mesh = trimesh.Trimesh(
                            vertices=gt_verts[-1].detach().cpu().numpy(),
                            faces=self._affine_mano.closed_faces.detach().cpu().numpy(),
                        )
                        visualize_MANO(
                            pred_hand_mesh,
                            obj_mesh=batch_obj_data["mesh"][0],
                            gt_hand=gt_hand_mesh,
                        )
                        visualize_MANO(
                            pred_hand_mesh,
                            obj_mesh=batch_obj_data["mesh"][0],
                            opacity=1.0,
                        )
                        eval_metrics[
                            "With contact fitting"
                        ] = self._compute_eval_metrics(
                            anchors_pred,
                            verts_pred,
                            joints_pred,
                            gt_anchors,
                            gt_verts,
                            gt_joints,
                            batch_obj_data,
                        )
        return eval_metrics

    @to_cuda
    def _test_iteration(
        self,
        batch: Union[Tuple, List, torch.Tensor],
        n_observations: int,
        batch_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """Evaluation procedure for one batch. We want to keep the code DRY and avoid
        making mistakes, so this code calls the BaseTrainer._train_val_iteration() method.
        Args:
            batch: The batch to process.
        Returns:
            torch.Tensor: The loss for the batch.
        """
        samples, labels, mesh_pths = batch  # type: ignore
        eval_metrics = self._test_batch(
            samples,
            labels,
            mesh_pths,
            n_observations,
            batch_idx,
            use_prior=True,
        )
        # TODO: Refactor this mess! I can just return the right dict in _test_batch and get rid of
        # _test_iteration which is completely useless :/. There was a lot of bad code written in a
        # hurry before this, and little by little we ended up here. Research code amirite?!
        batch_metrics = {}
        for k, v in eval_metrics.items():
            if v is not None:
                for metric_name, metric_val in v.items():
                    batch_metrics[f"[{k}] {metric_name}"] = metric_val
        return batch_metrics

    @to_cuda
    def _save_batch_predictions_as_sequence(
        self,
        samples: Dict,
        labels: Dict,
        mesh_pths: List[str],
        n_observations: int,
        batch_idx: int,
        scenes: Dict,
    ):
        print(f"For {n_observations} observations")
        hand_color = trimesh.visual.random_color()
        input_scalar = samples["scalar"]
        if len(input_scalar.shape) == 2:
            input_scalar = input_scalar.mean(
                dim=1
            )  # TODO: Think of a better way for 'pair' scaling. Never mind we have object scaling which is better
        y_hat = self._inference(samples, labels, use_prior=True)
        mano_params_gt = {
            "pose": labels["theta"].view(-1, *labels["theta"].shape[2:]),
            "beta": labels["beta"].view(-1, *labels["beta"].shape[2:]),
            "rot_6d": labels["rot"].view(-1, *labels["rot"].shape[2:]),
            "trans": labels["trans"].view(-1, *labels["trans"].shape[2:]),
        }
        mano_params_input = {
            "pose": samples["theta"].view(-1, *samples["theta"].shape[2:]),
            "beta": samples["beta"].view(-1, *samples["beta"].shape[2:]),
            "rot_6d": samples["rot"].view(-1, *samples["rot"].shape[2:]),
            "trans": samples["trans"].view(-1, *samples["trans"].shape[2:]),
        }
        # Only use the last view for each batch element (they're all the same anyway for static
        # grasps, but for dynamic grasps we want to predict the LAST frame!).
        # mano_params_gt = {k: v for k, v in mano_params_gt.items()}
        gt_verts, _ = self._affine_mano(*mano_params_gt.values())
        sample_verts, sample_joints = self._affine_mano(*mano_params_input.values())
        if not self._data_loader.dataset.is_right_hand_only:
            raise NotImplementedError("Right hand only is implemented for testing.")
        multiple_obs = len(samples["theta"].shape) > 2
        # For mesh_pths we have a tuple of N lists of B entries. N is the number of
        # observations and B is the batch size. We'll take the last observation for each batch
        # element.
        mesh_pths_iter = mesh_pths[-1]  # Now we have a list of B entries.
        mesh = os.path.basename(mesh_pths_iter[0]).split(".")[0]
        if mesh in [
            "wineglass",
            "binoculars",
            "camera",
            "mug",
            "fryingpan",
            "toothpaste",
        ]:
            return scenes
        print(f"Rendering {mesh}")
        use_smplx = False  # TODO: I don't use it for now

        with torch.set_grad_enabled(True):
            (
                _,
                _,
                _,
                _,
                anchors_pred,
                verts_pred,
                joints_pred,
            ) = optimize_pose_pca_from_choir(
                y_hat["choir"],
                bps=self._bps,
                anchor_indices=self._anchor_indices,
                scalar=input_scalar,
                max_iterations=2000,
                loss_thresh=1e-6,
                lr=8e-2,
                is_rhand=samples["is_rhand"],
                use_smplx=use_smplx,
                dataset=self._data_loader.dataset.name,
                remap_bps_distances=self._remap_bps_distances,
                exponential_map_w=self._exponential_map_w,
                initial_params={
                    k: (
                        v[:, -1] if multiple_obs else v
                    )  # Initial pose is the last observation
                    for k, v in samples.items()
                    if k
                    in ["theta", ("vtemp" if use_smplx else "beta"), "rot", "trans"]
                },
                beta_w=1e-4,
                theta_w=1e-7,
                choir_w=1000,
            )
            image_dir = os.path.join(
                HydraConfig.get().runtime.output_dir,
                "tto_images",
            )
            if not os.path.exists(image_dir):
                os.makedirs(image_dir)
            viewed_meshes = []
            obj_meshes = {}
            for i, mesh_pth in enumerate(mesh_pths_iter):
                mesh_name = os.path.basename(mesh_pth)
                pred_hand_mesh = trimesh.Trimesh(
                    vertices=verts_pred[i].detach().cpu().numpy(),
                    faces=self._affine_mano.closed_faces.detach().cpu().numpy(),
                )
                pred_hand_mesh.visual.vertex_colors = hand_color
                # i corresponds to batch element
                # sample_verts is (B, T, V, 3) but (B*T, V, 3) actually. So to index [i, n-1] we
                # need to do [i * T + n - 1]. n-1 because n is 1-indexed.
                input_hand_mesh = trimesh.Trimesh(
                    vertices=sample_verts[
                        i * samples["theta"].shape[1] + n_observations - 1
                    ]
                    .detach()
                    .cpu()
                    .numpy(),
                    faces=self._affine_mano.closed_faces.detach().cpu().numpy(),
                )
                input_hand_mesh.visual.vertex_colors = hand_color
                gt_hand_mesh = trimesh.Trimesh(
                    vertices=gt_verts[i * labels["theta"].shape[1] + n_observations - 1]
                    .detach()
                    .cpu()
                    .numpy(),
                    faces=self._affine_mano.closed_faces.detach().cpu().numpy(),
                )
                gt_hand_mesh.visual.vertex_colors = hand_color
                if mesh_name not in obj_meshes:
                    obj_mesh = o3dio.read_triangle_mesh(mesh_pth)
                    if self._data_loader.dataset.center_on_object_com:
                        obj_mesh.translate(-obj_mesh.get_center())
                    obj_mesh = Trimesh(obj_mesh.vertices, obj_mesh.triangles)
                    obj_mesh.visual.vertex_colors = trimesh.visual.random_color()
                    obj_meshes[mesh_name] = obj_mesh

                if mesh_name not in scenes:
                    scene_anim = ScenePicAnim()
                    scenes[mesh_name] = scene_anim

                scenes[mesh_name].add_frame(
                    {
                        "object": obj_mesh,
                        "hand": pred_hand_mesh,
                        "hand_aug": input_hand_mesh,
                        "gt_hand": gt_hand_mesh,
                    }
                )
            return scenes

    @to_cuda
    def _save_batch_predictions(
        self,
        samples: Dict,
        labels: Dict,
        mesh_pths: List[str],
        n_observations: int,
        batch_idx: int,
    ):
        plots = {}
        gt_is_plotted = False
        batch_obj_data = make_batch_of_obj_data(self._object_cache, mesh_pths)
        for n in range(1, n_observations + 1):
            print(f"For {n} observations")
            print(samples["choir"].shape, samples["choir"][:, :n].shape)
            input_scalar = samples["scalar"]
            if len(input_scalar.shape) == 2:
                input_scalar = input_scalar.mean(
                    dim=1
                )  # TODO: Think of a better way for 'pair' scaling. Never mind we have object scaling which is better
            # y_hat = self._model(samples["choir"][:, :n], use_mean=True)
            y_hat = self._inference(
                samples, labels, use_prior=True, max_observations=n_observations
            )
            print(y_hat["choir"].shape)
            mano_params_gt = {
                "pose": labels["theta"][:, :n],
                "beta": labels["beta"][:, :n],
                "rot_6d": labels["rot"][:, :n],
                "trans": labels["trans"][:, :n],
            }
            mano_params_input = {
                "pose": samples["theta"].view(-1, *samples["theta"].shape[2:]),
                "beta": samples["beta"].view(-1, *samples["beta"].shape[2:]),
                "rot_6d": samples["rot"].view(-1, *samples["rot"].shape[2:]),
                "trans": samples["trans"].view(-1, *samples["trans"].shape[2:]),
            }
            # Only use the last view for each batch element (they're all the same anyway for static
            # grasps, but for dynamic grasps we want to predict the LAST frame!).
            mano_params_gt = {k: v[:, -1] for k, v in mano_params_gt.items()}
            gt_pose, gt_shape, gt_rot_6d, gt_trans = tuple(mano_params_gt.values())
            gt_verts, gt_joints = self._affine_mano(
                gt_pose, gt_shape, gt_rot_6d, gt_trans
            )
            sample_verts, sample_joints = self._affine_mano(*mano_params_input.values())
            if not self._data_loader.dataset.is_right_hand_only:
                raise NotImplementedError("Right hand only is implemented for testing.")
            multiple_obs = len(samples["theta"].shape) > 2
            # For mesh_pths we have a tuple of N lists of B entries. N is the number of
            # observations and B is the batch size. We'll take the last observation for each batch
            # element.
            mesh_pths_iter = mesh_pths[-1]  # Now we have a list of B entries.
            use_smplx = False  # TODO: I don't use it for now
            contacts_pred, obj_points, obj_normals = (
                y_hat.get("contacts", None),
                None,
                None,
            )
            if self._enable_contacts_tto and contacts_pred is not None:
                obj_points, obj_normals = (
                    batch_obj_data["points"],
                    batch_obj_data["normals"],
                )
            with torch.set_grad_enabled(True):
                (
                    _,
                    _,
                    _,
                    _,
                    anchors_pred,
                    verts_pred,
                    joints_pred,
                ) = optimize_pose_pca_from_choir(
                    y_hat["choir"],
                    contact_gaussians=y_hat.get("contacts", None),
                    obj_pts=obj_points,
                    obj_normals=obj_normals,
                    bps=self._bps,
                    anchor_indices=self._anchor_indices,
                    scalar=input_scalar,
                    max_iterations=2000,
                    loss_thresh=1e-6,
                    lr=8e-2,
                    is_rhand=samples["is_rhand"],
                    use_smplx=use_smplx,
                    dataset=self._data_loader.dataset.name,
                    remap_bps_distances=self._remap_bps_distances,
                    exponential_map_w=self._exponential_map_w,
                    initial_params={
                        k: (
                            v[:, :n][:, -1] if multiple_obs else v[:, :n]
                        )  # Initial pose is the last observation
                        for k, v in samples.items()
                        if k
                        in ["theta", ("vtemp" if use_smplx else "beta"), "rot", "trans"]
                    },
                    beta_w=1e-4,
                    theta_w=1e-7,
                    choir_w=1000,
                )
                image_dir = os.path.join(
                    HydraConfig.get().runtime.output_dir,
                    "tto_images",
                )
                if not os.path.exists(image_dir):
                    os.makedirs(image_dir)
                viewed_meshes = []
                pyvista_obj_meshes, hands_trimesh = {}, {}
                for i, mesh_pth in enumerate(mesh_pths_iter):
                    if mesh_pth in hands_trimesh:
                        pred_hand_mesh = hands_trimesh[mesh_pth]
                    else:
                        pred_hand_mesh = trimesh.Trimesh(
                            vertices=verts_pred[i].detach().cpu().numpy(),
                            faces=self._affine_mano.closed_faces.detach().cpu().numpy(),
                        )
                        hands_trimesh[mesh_pth] = pred_hand_mesh
                    if mesh_pth in pyvista_obj_meshes:
                        obj_mesh_pv = pyvista_obj_meshes[mesh_pth]
                    else:
                        obj_mesh = self._object_cache[os.path.basename(mesh_pth)][
                            "mesh"
                        ]
                        obj_mesh_pv = pv.wrap(obj_mesh)
                        pyvista_obj_meshes[mesh_pth] = obj_mesh_pv

                    grasp_key = (mesh_name, i)
                    if grasp_key not in plots:
                        pl = pv.Plotter(
                            shape=(2, n_observations + 1),
                            border=False,
                            off_screen=False,
                        )
                        plots[grasp_key] = pl
                        with open(
                            os.path.join(
                                image_dir, f"{mesh_name}_{i}_tto_{batch_idx}.json"
                            ),
                            "w",
                        ) as f:
                            f.write(
                                json.dumps(
                                    {
                                        "beta": mano_params_input["beta"][
                                            i * samples["beta"].shape[1] + n - 1
                                        ]
                                        .detach()
                                        .cpu()
                                        .numpy()
                                        .tolist(),
                                        "pose": mano_params_input["pose"][
                                            i * samples["theta"].shape[1] + n - 1
                                        ]
                                        .detach()
                                        .cpu()
                                        .numpy()
                                        .tolist(),
                                        "trans": mano_params_input["trans"][
                                            i * samples["trans"].shape[1] + n - 1
                                        ]
                                        .detach()
                                        .cpu()
                                        .numpy()
                                        .tolist(),
                                        "hTm": rotation_6d_to_matrix(
                                            mano_params_input["rot_6d"][
                                                i * samples["rot"].shape[1] + n - 1
                                            ].detach()
                                        )
                                        .cpu()
                                        .numpy()
                                        .tolist(),
                                    }
                                )
                            )

                    pl = plots[grasp_key]
                    if not gt_is_plotted:
                        pl.subplot(0, 0)
                        gt_hand_mesh = trimesh.Trimesh(
                            vertices=gt_verts[i].detach().cpu().numpy(),
                            faces=self._affine_mano.closed_faces.detach().cpu().numpy(),
                        )
                        gt_hand = pv.wrap(gt_hand_mesh)
                        pl.add_mesh(
                            gt_hand,
                            opacity=1.0,
                            name="gt_hand",
                            label="Ground-truth Hand",
                            smooth_shading=True,
                        )
                        pl.add_mesh(
                            obj_mesh_pv,
                            opacity=1.0,
                            name="obj_mesh",
                            label="Object mesh",
                            smooth_shading=True,
                            color="red",
                        )
                        pl.subplot(1, n)
                        # i corresponds to batch element
                        # sample_verts is (B, T, V, 3) but (B*T, V, 3) actually. So to index [i, n-1] we need to do [i * T + n - 1]. n-1 because n is 1-indexed.
                        input_hand_mesh = trimesh.Trimesh(
                            vertices=sample_verts[i * samples["theta"].shape[1] + n - 1]
                            .detach()
                            .cpu()
                            .numpy(),
                            faces=self._affine_mano.closed_faces.detach().cpu().numpy(),
                        )
                        input_hand = pv.wrap(input_hand_mesh)
                        pl.add_mesh(
                            input_hand,
                            opacity=1.0,
                            name="input_hand",
                            label="Input Hand",
                            smooth_shading=True,
                        )
                        pl.add_mesh(
                            obj_mesh_pv,
                            opacity=1.0,
                            name="obj_mesh",
                            label="Object mesh",
                            smooth_shading=True,
                            color="red",
                        )

                    pl.subplot(0, n)
                    hand_mesh = pv.wrap(pred_hand_mesh)
                    pl.add_mesh(
                        hand_mesh,
                        opacity=1.0,
                        name="hand_mesh",
                        label="Predicted Hand",
                        smooth_shading=True,
                    )
                    pl.add_mesh(
                        obj_mesh_pv,
                        opacity=1.0,
                        name="obj_mesh",
                        label="Object mesh",
                        smooth_shading=True,
                        color="red",
                    )
                    viewed_meshes.append(mesh_name)
        for (mesh_name, i), plot in plots.items():
            plot.set_background("white")  # type: ignore
            plot.link_views()
            plot.export_html(
                os.path.join(image_dir, f"{mesh_name}_{i}_tto_{batch_idx}.html")
            )

    def _save_model_predictions(self, n_observations: int) -> None:
        # Use a batch size of 1 cause no clue what happens above that
        """The deadline is in 20h so I'll write this quick and dirty, sorry reader."""
        self._data_loader.dataset.set_observations_number(n_observations)
        self._pbar = tqdm(total=len(self._data_loader), desc="Exporting predictions")
        self._pbar.refresh()
        scenes = {}
        for i, batch in enumerate(self._data_loader):
            if not self._running:
                print("[!] Testing aborted.")
            samples, labels, mesh_pths = batch  # type: ignore
            if self._data_loader.dataset.name.lower() == "contactpose":
                self._save_batch_predictions(
                    samples, labels, mesh_pths, n_observations, i
                )
            elif self._data_loader.dataset.name.lower() == "grab":
                scenes = self._save_batch_predictions_as_sequence(
                    samples, labels, mesh_pths, n_observations, i, scenes
                )
            if len(list(scenes.keys())) > 1:
                del scenes[list(scenes.keys())[-1]]
                break
            if (
                len(list(scenes.keys())) > 0
                and scenes[list(scenes.keys())[0]].n_frames > 130
            ):
                break
            self._pbar.update()
        for mesh_name, scene in scenes.items():
            print(f"Saving {mesh_name}")
            scene.save_animation(f"{mesh_name}_tto.html")
        self._pbar.close()

    def test_n_observations(
        self,
        n_observations: int,
        visualize_every: int = 0,
    ) -> Dict[str, float]:
        metrics = defaultdict(MeanMetric)
        color_code = project_conf.ANSI_COLORS[project_conf.Theme.TESTING.value]
        self._data_loader.dataset.set_observations_number(n_observations)
        self._pbar = tqdm(
            total=len(self._data_loader), desc=f"Testing {n_observations} observations"
        )
        self._pbar.refresh()
        for i, batch in enumerate(self._data_loader):
            if not self._running:
                print("[!] Testing aborted.")
                break
            batch_metrics = self._test_iteration(batch, n_observations, i)
            for k, v in batch_metrics.items():
                metrics[k].update(v)
            del batch_metrics
            " ==================== Visualization ==================== "
            if visualize_every > 0 and (i + 1) % visualize_every == 0:
                self._visualize(batch, color_code)
            self._pbar.update()
        self._pbar.close()
        print("=" * 81)
        print(
            "==" + " " * 28 + f" Test results (N={n_observations}) " + " " * 28 + "=="
        )
        print("=" * 81)
        # Compute all metrics:
        computed_metrics = {}
        for k, v in metrics.items():
            computed_metrics[k] = v.compute().item()
            print(f"\t -> {k}: {computed_metrics[k]:.2f}")
        print("_" * 81)
        return computed_metrics

    def test(self, visualize_every: int = 0):
        """Computes the average loss on the test set.
        Args:
            visualize_every (int, optional): Visualize the model predictions every n batches.
            Defaults to 0 (no visualization).
        """
        self._model.eval()
        if self._save_predictions:
            with torch.no_grad():
                self._save_model_predictions(
                    min(4, self._max_observations)
                    if self._data_loader.dataset.name.lower() == "contactpose"
                    else 7
                )
            return
        test_errors = []
        with torch.no_grad():
            for i in range(1, self._max_observations + 1):
                test_errors.append(
                    self.test_n_observations(
                        i,
                        visualize_every=visualize_every,
                    )
                )

        with open(f"test_errors_{self._run_name}.pickle", "wb") as f:
            compressed_pkl = blosc2.compress(pickle.dumps(test_errors))
            f.write(compressed_pkl)

        return

        # Plot a curve of the test errors and compute Area Under Curve (AUC). Add marker for the
        # reported ContactPose results of (25.05mm and replicate the results of the paper to have a
        # fair comparison with my different additive noise for Perturbed ContactPose).
        plt.figure(figsize=(8, 6))
        plt.plot(
            [1, 2, 3, 4],
            [x["[PRIOR] MPJPE (mm)"] for x in test_errors],
            "-o",
            label="MPJPE (mm)",
        )
        plt.plot(
            [1, 2, 3, 4],
            [x["[PRIOR] Root-aligned MPJPE (mm)"] for x in test_errors],
            "-o",
            label="Root-aligned MPJPE (mm)",
        )
        # ContactOpt reported an absolute MPJPE of 25.05mm:
        plt.plot(
            [1],
            [25.05],
            "*",
            label="ContactOpt",
        )
        # Compute the AUC for the MPJPE curve:
        # auc = torch.trapz(
        # torch.tensor([x["[PRIOR] MPJPE (mm)"] for x in test_errors]),
        # torch.tensor([1, 2, 3, 4]),
        # )
        plt.title(f"Test MPJPE and Root-aligned MPJPE for N observations")
        # Add the AUC to the legend:
        # plt.legend(title=f"MPJPE AUC: {auc:.2f}")
        plt.legend(title=f"Legend")
        plt.xlabel("N observations")
        plt.ylabel("(Root-aligned) MPJPE (mm)")
        plt.savefig(f"test_error_{self._run_name}.png")
        plt.show()

        plt.figure(figsize=(8, 6))
        plt.plot(
            [1, 2, 3, 4],
            [x["[PRIOR] MPJPE (mm)"] for x in test_errors],
            "-o",
            label="MPJPE (mm)",
        )

        # ContactOpt reported an absolute MPJPE of 25.05mm:
        plt.plot(
            [1],
            [25.05],
            "*",
            label="ContactOpt",
        )
        # Compute the AUC for the MPJPE curve:
        # auc = torch.trapz(
        # torch.tensor([x["[PRIOR] MPJPE (mm)"] for x in test_errors]),
        # torch.tensor([1, 2, 3, 4]),
        # )
        plt.title(f"Test MPJPE for N observations")
        # Add the AUC to the legend:
        # plt.legend(title=f"MPJPE AUC: {auc:.2f}")
        plt.legend(title=f"Legend")
        plt.xlabel("N observations")
        plt.ylabel("MPJPE (mm)")
        plt.savefig(f"test_error_mpjpe_only_{self._run_name}.png")
        plt.show()
