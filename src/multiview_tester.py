#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.


import os
import pickle
import signal
from collections import defaultdict
from typing import Dict, List, Tuple, Union

import blosc
import matplotlib.pyplot as plt
import numpy as np
import open3d.io as o3dio
import torch
import trimesh
import trimesh.voxel.creation as voxel_create
from hydra.core.hydra_config import HydraConfig
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric
from tqdm import tqdm

from conf import project as project_conf
from model.affine_mano import AffineMANO
from src.multiview_trainer import MultiViewTrainer
from utils import colorize, to_cuda, to_cuda_
from utils.training import optimize_pose_pca_from_choir
from utils.visualization import (
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
    ) -> None:
        """Base trainer class.
        Args:
            model (torch.nn.Module): Model to train.
            opt (torch.optim.Optimizer): Optimizer to use.
            data_loader (torch.utils.data.DataLoader): Training dataloader.
            val_loader (torch.utils.data.DataLoader): Validation dataloader.
        """
        self._run_name = run_name
        self._model = model
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
        self._pbar = tqdm(total=1, desc="Testing")
        self._affine_mano = to_cuda_(
            AffineMANO(
                ncomps=self._data_loader.dataset.theta_dim - 3,
                flat_hand_mean=(self._data_loader.dataset.name == "grab"),
                for_contactpose=(self._data_loader.dataset.name == "contactpose"),
            )
        )
        self._bps_dim = data_loader.dataset.bps_dim
        self._bps = to_cuda_(data_loader.dataset.bps)
        self._anchor_indices = to_cuda_(data_loader.dataset.anchor_indices)
        self._remap_bps_distances = data_loader.dataset.remap_bps_distances
        self._exponential_map_w = data_loader.dataset.exponential_map_w
        self._n_ctrl_c = 0
        signal.signal(signal.SIGINT, self._terminator)

    @to_cuda
    def _visualize(
        self,
        batch: Union[Tuple, List, torch.Tensor],
        epoch: int,
    ) -> None:
        """Visualize the model predictions.
        Args:
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

    def _test_batch(
        self,
        samples: Dict,
        labels: Dict,
        mesh_pths: List[str],
        n_observations: int,
        batch_idx: int,
        use_prior: bool = True,
        use_input: bool = False,
        save_predictions: bool = False,
    ) -> Tuple:
        input_scalar = samples["scalar"]
        if len(input_scalar.shape) == 2:
            input_scalar = input_scalar.mean(
                dim=1
            )  # TODO: Think of a better way for 'pair' scaling. Never mind we have object scaling which is better
        if use_prior:
            y_hat = self._model(samples["choir"], use_mean=True)
        else:
            y_hat = self._model(samples["choir"], labels["choir"], use_mean=True)
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
        gt_verts, gt_joints = self._affine_mano(gt_pose, gt_shape, gt_rot_6d, gt_trans)
        gt_anchors = self._affine_mano.get_anchors(gt_verts)
        if not self._data_loader.dataset.is_right_hand_only:
            raise NotImplementedError("Right hand only is implemented for testing.")
        multiple_obs = len(samples["theta"].shape) > 2
        # For mesh_pths we have a tuple of N lists of B entries. N is the number of
        # observations and B is the batch size. We'll take the last observation for each batch
        # element.
        mesh_pths = mesh_pths[-1]  # Now we have a list of B entries.

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
        else:
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
            if save_predictions:
                image_dir = os.path.join(
                    HydraConfig.get().runtime.output_dir,
                    "tto_images",
                    f"N={n_observations}",
                )
                if not os.path.exists(image_dir):
                    os.makedirs(image_dir)
                viewed_meshes = []
                for i, mesh_pth in enumerate(mesh_pths):
                    mesh_name = os.path.basename(mesh_pth)
                    if mesh_name not in viewed_meshes:
                        pred_hand_mesh = trimesh.Trimesh(
                            vertices=verts_pred[i].detach().cpu().numpy(),
                            faces=self._affine_mano.closed_faces.detach().cpu().numpy(),
                        )
                        gt_hand_mesh = trimesh.Trimesh(
                            vertices=gt_verts[i].detach().cpu().numpy(),
                            faces=self._affine_mano.closed_faces.detach().cpu().numpy(),
                        )
                        obj_mesh = o3dio.read_triangle_mesh(mesh_pth)
                        if self._data_loader.dataset.center_on_object_com:
                            obj_mesh.translate(-obj_mesh.get_center())
                        cam_pose = visualize_MANO(
                            gt_hand_mesh,
                            obj_mesh,
                            save_as=os.path.join(
                                image_dir, f"{mesh_name}_tto_{batch_idx}_GT.html"
                            ),
                            opacity=1.0,
                            return_cam_pose=True,
                        )
                        visualize_MANO(
                            pred_hand_mesh,
                            obj_mesh,
                            save_as=os.path.join(
                                image_dir, f"{mesh_name}_tto_{batch_idx}.html"
                            ),
                            opacity=1.0,
                            cam_pose=cam_pose,
                        )
                        viewed_meshes.append(mesh_name)
        with torch.no_grad():
            # === Anchor error ===
            anchor_error = (
                torch.norm(anchors_pred - gt_anchors.cuda(), dim=2)
                .mean(dim=1)
                .mean(dim=0)
                * self._data_loader.dataset.base_unit
            )
            # === MPJPE ===
            pjpe = torch.linalg.vector_norm(
                gt_joints - joints_pred, ord=2, dim=-1
            )  # Per-joint position error (B, 21)
            mpjpe = torch.mean(pjpe, dim=-1)  # Mean per-joint position error (B, 1)
            mpjpe = torch.mean(
                mpjpe, dim=0
            )  # Mean per-joint position error avgd across batch (1)
            mpjpe *= self._data_loader.dataset.base_unit
            root_aligned_pjpe = torch.linalg.vector_norm(
                (gt_joints - gt_joints[:, 0:1, :])
                - (joints_pred - joints_pred[:, 0:1, :]),
                ord=2,
                dim=-1,
            )  # Per-joint position error (B, 21)
            root_aligned_mpjpe = torch.mean(
                root_aligned_pjpe, dim=-1
            )  # Mean per-joint position error (B, 1)
            root_aligned_mpjpe = torch.mean(
                root_aligned_mpjpe, dim=0
            )  # Mean per-joint position error avgd across batch (1)
            root_aligned_mpjpe *= self._data_loader.dataset.base_unit
            # ====== MPVPE ======
            # Compute the mean per-vertex position error (MPVPE) between the predicted and ground truth
            # hand meshes.
            pvpe = torch.linalg.vector_norm(
                gt_verts - verts_pred, ord=2, dim=-1
            )  # Per-vertex position error (B, 778, 3)
            mpvpe = torch.mean(pvpe, dim=-1)  # Mean per-vertex position error (B, 1)
            mpvpe = torch.mean(
                mpvpe, dim=0
            )  # Mean per-vertex position error avgd across batch (1)
            mpvpe *= self._data_loader.dataset.base_unit
            # ====== Intersection volume ======
            # Compute the intersection volume between the predicted hand meshes and the object meshes.
            # TODO: Implement it with another method cause it crashes (see
            # https://github.com/isl-org/Open3D/issues/5911)
            # intersection_volumes = []
            # mano_faces = self._affine_mano.faces.cpu().numpy()
            # for i, path in enumerate(mesh_pths):
            # obj_mesh = o3dtg.TriangleMesh.from_legacy(o3dio.read_triangle_mesh(path))
            # hand_mesh = o3dtg.TriangleMesh.from_legacy(
            # o3dg.TriangleMesh(
            # o3du.Vector3dVector(verts_pred[i].cpu().numpy()),
            # o3du.Vector3iVector(mano_faces),
            # )
            # )
            # intersection = obj_mesh.boolean_intersection(hand_mesh)
            # intersection_volumes.append(intersection.to_legacy().get_volume())
            # intersection_volume = torch.tensor(intersection_volumes).mean()
            # intersection_volume *= (
            # self._data_loader.dataset.base_unit / 10
            # )  # m^3 -> mm^3 -> cm^3
            # We'll do the same with trimesh.boolean.intersection():
            # intersection_volumes = []
            # mano_faces = self._affine_mano.closed_faces.cpu().numpy()
            # print("[*] Computing intersection volumes...")
            # for i, path in tqdm(enumerate(mesh_pths), total=len(mesh_pths)):
            # obj_mesh = o3dio.read_triangle_mesh(path)
            # if self._data_loader.dataset.center_on_object_com:
            # obj_mesh.translate(-obj_mesh.get_center())
            # obj_mesh = trimesh.Trimesh(
            # vertices=obj_mesh.vertices, faces=obj_mesh.triangles
            # )
            # hand_mesh = trimesh.Trimesh(verts_pred[i].cpu().numpy(), mano_faces)
            # # TODO: Recenter the obj_mesh and potentially rescale it. Visualize to check.
            # # visualize_MANO(hand_mesh, obj_mesh=obj_mesh)
            # intersection = obj_mesh.intersection(hand_mesh)
            # if intersection.volume > 0:
            # print(self._data_loader.dataset.base_unit)
            # print(f"{intersection.volume * (self._data_loader.dataset.base_unit/10):.2f} cm^3")
            # print(f"Is water tight? {intersection.is_watertight}")
            # visualize_MANO(hand_mesh, obj_mesh=obj_mesh)
            # visualize_MANO(hand_mesh, obj_mesh=intersection)
            # intersection_volumes.append(intersection.volume)

            # Let's now try by voxelizing the meshes and reporting the volume of voxels occupied by
            # both meshes:
            intersection_volumes = []
            mano_faces = self._affine_mano.faces.cpu().numpy()
            pitch_mm = 2
            pitch = pitch_mm / self._data_loader.dataset.base_unit  # mm -> m
            # TODO: The radius only needs to be slightly larger than the object bounding box.
            radius = int(0.2 / pitch)  # 20cm in each direction for the voxel grid
            # TODO: This PoC works, but I need to make sure that the object mesh is always in its
            # canonical position in the test set! This might be the case for ContactPose, but probably
            # not for GRAB.
            # TODO: Multithread this?
            # TODO: Do the same with PyVista or Open3D? Whatever is fastest because this is slooooow.
            obj_voxels = {}
            for i, path in tqdm(enumerate(mesh_pths), total=len(mesh_pths)):
                if path not in obj_voxels:
                    obj_mesh = o3dio.read_triangle_mesh(path)
                    if self._data_loader.dataset.center_on_object_com:
                        obj_mesh.translate(-obj_mesh.get_center())

                    obj_mesh = trimesh.Trimesh(
                        vertices=obj_mesh.vertices, faces=obj_mesh.triangles
                    )
                    obj_voxel = (
                        voxel_create.local_voxelize(
                            obj_mesh, np.array([0, 0, 0]), pitch, radius
                        )
                        .fill()
                        .matrix
                    )
                    obj_voxels[path] = obj_voxel
                else:
                    obj_voxel = obj_voxels[path]
                hand_mesh = trimesh.Trimesh(verts_pred[i].cpu().numpy(), mano_faces)
                hand_voxel = (
                    voxel_create.local_voxelize(
                        hand_mesh, np.array([0, 0, 0]), pitch, radius
                    )
                    .fill()
                    .matrix
                )
                # both_voxels = trimesh.voxel.VoxelGrid(
                # trimesh.voxel.encoding.DenseEncoding(
                # obj_voxel | hand_voxel
                # ),
                # )
                # both_voxels.show()
                # obj_volume = (
                # np.count_nonzero(obj_voxel) * (pitch**3) * 1000000
                # )  # m^3 -> cm^3
                # hand_volume = (
                # np.count_nonzero(hand_voxel) * (pitch**3) * 1000000
                # )  # m^3 -> cm^3
                # typical_hand_volume = (
                # 379.7  # cm^3 https://doi.org/10.1177/154193128603000417
                # )
                # Make sure we're within 35% of the typical hand volume:
                # assert (
                # hand_volume > typical_hand_volume * 0.65
                # and hand_volume < typical_hand_volume * 1.35
                # ), f"Hand volume is {hand_volume:.2f} cm^3, which is not within 30% of the typical"
                # if (
                # hand_volume > typical_hand_volume * 0.65
                # and hand_volume < typical_hand_volume * 1.35
                # ):
                # print(
                # f"Hand volume is {hand_volume:.2f} cm^3, which is not within 30% of the typical"
                # )
                intersection_volume = (
                    np.count_nonzero((obj_voxel & hand_voxel)) * (pitch**3) * 1000000
                )
                # print(
                # f"Volume of hand: {hand_volume:.2f} cm^3, volume of obj: {obj_volume:.2f} cm^3, intersection volume: {intersection_volume:.2f} cm^3"
                # )
                intersection_volumes.append(intersection_volume)

        intersection_volume = torch.tensor(intersection_volumes).float().mean()
        return (
            anchor_error.detach(),
            mpjpe.detach(),
            root_aligned_mpjpe.detach(),
            mpvpe.detach(),
            intersection_volume.detach(),
        )

    @to_cuda
    def _test_iteration(
        self,
        batch: Union[Tuple, List, torch.Tensor],
        n_observations: int,
        batch_idx: int,
        save_predictions: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Evaluation procedure for one batch. We want to keep the code DRY and avoid
        making mistakes, so this code calls the BaseTrainer._train_val_iteration() method.
        Args:
            batch: The batch to process.
        Returns:
            torch.Tensor: The loss for the batch.
        """
        samples, labels, mesh_pths = batch  # type: ignore
        # (
        # anchor_error_x,
        # mpjpe_x,
        # root_aligned_mpjpe_x,
        # mpvpe_x,
        # intesection_volume_x,
        # ) = self._test_batch(samples, labels, mesh_pths, n_observations, batch_idx, use_input=True)
        (
            anchor_error_p,
            mpjpe_p,
            root_aligned_mpjpe_p,
            mpvpe_p,
            intersection_volume_p,
        ) = self._test_batch(
            samples,
            labels,
            mesh_pths,
            n_observations,
            batch_idx,
            use_prior=True,
            save_predictions=save_predictions,
        )
        # (
        # anchor_error,
        # mpjpe,
        # root_aligned_mpjpe,
        # mpvpe,
        # intersection_volume,
        # ) = self._test_batch(
        # samples, labels, mesh_pths, n_observations, batch_idx, use_prior=False
        # )
        return {
            "[PRIOR] Anchor error (mm)": anchor_error_p,
            "[PRIOR] MPJPE (mm)": mpjpe_p,
            "[PRIOR] Root-aligned MPJPE (mm)": root_aligned_mpjpe_p,
            "[PRIOR] MPVPE (mm)": mpvpe_p,
            "[PRIOR] Intersection volume (cm3)": intersection_volume_p,
            # "[POSTERIOR] Anchor error (mm)": anchor_error,
            # "[POSTERIOR] MPJPE (mm)": mpjpe,
            # "[POSTERIOR] Root-aligned MPJPE (mm)": root_aligned_mpjpe,
            # "[POSTERIOR] MPVPE (mm)": mpvpe,
            # "[POSTERIOR] Intersection volume (cm3)": intersection_volume,
            # "[NOISY INPUT] Anchor error (mm)": anchor_error_x,
            # "[NOISY INPUT] MPJPE (mm)": mpjpe_x,
            # "[NOISY INPUT] Root-aligned MPJPE (mm)": root_aligned_mpjpe_x,
            # "[NOISY INPUT] MPVPE (mm)": mpvpe_x,
            # "[NOISY INPUT] Intersection volume (cm3)": intesection_volume_x,
        }

    def test_n_observations(
        self,
        n_observations: int,
        visualize_every: int = 0,
        save_predictions: bool = False,
    ) -> Dict[str, float]:
        metrics = defaultdict(MeanMetric)
        color_code = project_conf.ANSI_COLORS[project_conf.Theme.TESTING.value]
        self._data_loader.dataset.set_observations_number(n_observations)
        self._pbar = tqdm(total=len(self._data_loader), desc="Testing")
        self._pbar.refresh()
        for i, batch in enumerate(self._data_loader):
            if not self._running:
                print("[!] Testing aborted.")
                break
            batch_metrics = self._test_iteration(
                batch, n_observations, i, save_predictions
            )
            for k, v in batch_metrics.items():
                metrics[k].update(v.detach().item())
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

    def test(self, visualize_every: int = 0, **kwargs):
        """Computes the average loss on the test set.
        Args:
            visualize_every (int, optional): Visualize the model predictions every n batches.
            Defaults to 0 (no visualization).
        """
        self._model.eval()
        test_errors = []
        with torch.no_grad():
            for i in range(1, 15):
                test_errors.append(
                    self.test_n_observations(
                        i,
                        visualize_every=visualize_every,
                        save_predictions=kwargs["save_predictions"],
                    )
                )

        with open(f"test_errors_{self._run_name}.pickle", "wb") as f:
            compressed_pkl = blosc.compress(pickle.dumps(test_errors))
            f.write(compressed_pkl)

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
