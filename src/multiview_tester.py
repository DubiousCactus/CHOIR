#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.


import pickle
import signal
from collections import defaultdict
from typing import Dict, List, Tuple, Union

import blosc
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric
from tqdm import tqdm

from conf import project as project_conf
from model.affine_mano import AffineMANO
from src.multiview_trainer import MultiViewTrainer
from utils import to_cuda, to_cuda_
from utils.training import (
    get_dict_from_sample_and_label_tensors,
    optimize_pose_pca_from_choir,
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
        assert model_ckpt_path is not None, "No checkpoint path provided."
        self._load_checkpoint(model_ckpt_path, model_only=True)
        self._training_loss = training_loss
        self._data_loader = data_loader
        self._running = True
        self._pbar = tqdm(total=1, desc="Testing")
        self._affine_mano = to_cuda_(AffineMANO())
        self._bps_dim = data_loader.dataset.bps_dim
        self._bps = to_cuda_(data_loader.dataset.bps)
        self._remap_bps_distances = data_loader.dataset.remap_bps_distances
        self._exponential_map_w = data_loader.dataset.exponential_map_w
        signal.signal(signal.SIGINT, self._terminator)

    def _test_batch(
        self,
        samples: Dict,
        labels: Dict,
        mesh_pths: List[str],
        use_prior: bool = True,
        use_input: bool = False,
    ) -> Tuple:
        input_scalar = samples["scalar"]
        if len(input_scalar.shape) == 2 and not use_input:
            input_scalar = input_scalar.mean(
                dim=1
            )  # TODO: Think of a better way for 'pair' scaling. Never mind we have object scaling which is better
        elif len(input_scalar.shape) == 2 and use_input:
            input_scalar = input_scalar.view(-1)
        if use_input:
            B, N, P, D = samples["choir"].shape
            y_hat = {
                "choir": samples["choir"].view(B * N, P, D),
            }
        else:
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
        if use_input:
            # Use all views for each batch element
            mano_params_gt = {
                k: v.view(B * N, *v.shape[2:]) for k, v in mano_params_gt.items()
            }
        else:
            # Only use the first view for each batch element
            mano_params_gt = {k: v[:, 0] for k, v in mano_params_gt.items()}
        gt_pose, gt_shape, gt_rot_6d, gt_trans = tuple(mano_params_gt.values())
        gt_verts, gt_joints = self._affine_mano(gt_pose, gt_shape, gt_rot_6d, gt_trans)
        gt_anchors = self._affine_mano.get_anchors(gt_verts)
        if not self._data_loader.dataset.is_right_hand_only:
            raise NotImplementedError("Right hand only is implemented for testing.")
        with torch.set_grad_enabled(True):
            (
                pose,
                shape,
                rot_6d,
                trans,
                anchors_pred,
                verts_pred,
                joints_pred,
            ) = optimize_pose_pca_from_choir(
                y_hat["choir"],
                bps=self._bps,
                scalar=input_scalar,
                max_iterations=80,
                loss_thresh=1e-10,
                lr=4e-2,
                is_rhand=False,  # TODO
                use_smplx=False,  # TODO
                remap_bps_distances=self._remap_bps_distances,
                exponential_map_w=self._exponential_map_w,
            )
            # verts_pred, joints_pred = self._affine_mano(pose, shape, rot_6d, trans)
        # === Anchor error ===
        anchor_error = (
            torch.norm(anchors_pred - gt_anchors.cuda(), dim=2).mean(dim=1).mean(dim=0)
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
            (gt_joints - gt_joints[:, 0:1, :]) - (joints_pred - joints_pred[:, 0:1, :]),
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
        intersection_volume = torch.tensor(0.0)
        return anchor_error, mpjpe, root_aligned_mpjpe, mpvpe, intersection_volume

    @to_cuda
    def _test_iteration(
        self,
        batch: Union[Tuple, List, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Evaluation procedure for one batch. We want to keep the code DRY and avoid
        making mistakes, so this code calls the BaseTrainer._train_val_iteration() method.
        Args:
            batch: The batch to process.
        Returns:
            torch.Tensor: The loss for the batch.
        """
        x, y, mesh_pths = batch  # type: ignore
        samples, labels = get_dict_from_sample_and_label_tensors(x, y)
        (
            anchor_error_x,
            mpjpe_x,
            root_aligned_mpjpe_x,
            mpvpe_x,
            intesection_volume_x,
        ) = self._test_batch(samples, labels, mesh_pths, use_input=True)
        (
            anchor_error_p,
            mpjpe_p,
            root_aligned_mpjpe_p,
            mpvpe_p,
            intersection_volume_p,
        ) = self._test_batch(samples, labels, mesh_pths, use_prior=True)
        (
            anchor_error,
            mpjpe,
            root_aligned_mpjpe,
            mpvpe,
            intersection_volume,
        ) = self._test_batch(samples, labels, mesh_pths, use_prior=False)
        return {
            "[PRIOR] Anchor error (mm)": anchor_error_p,
            "[PRIOR] MPJPE (mm)": mpjpe_p,
            "[PRIOR] Root-aligned MPJPE (mm)": root_aligned_mpjpe_p,
            "[PRIOR] MPVPE (mm)": mpvpe_p,
            "[PRIOR] Intersection volume (cm3)": intersection_volume_p,
            "[POSTERIOR] Anchor error (mm)": anchor_error,
            "[POSTERIOR] MPJPE (mm)": mpjpe,
            "[POSTERIOR] Root-aligned MPJPE (mm)": root_aligned_mpjpe,
            "[POSTERIOR] MPVPE (mm)": mpvpe,
            "[POSTERIOR] Intersection volume (cm3)": intersection_volume,
            "[NOISY INPUT] Anchor error (mm)": anchor_error_x,
            "[NOISY INPUT] MPJPE (mm)": mpjpe_x,
            "[NOISY INPUT] Root-aligned MPJPE (mm)": root_aligned_mpjpe_x,
            "[NOISY INPUT] MPVPE (mm)": mpvpe_x,
            "[NOISY INPUT] Intersection volume (cm3)": intesection_volume_x,
        }

    def test_n_observations(
        self, n_observations: int, visualize_every: int = 0
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
            batch_metrics = self._test_iteration(batch)
            for k, v in batch_metrics.items():
                metrics[k].update(v.item())
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
            test_errors.append(
                self.test_n_observations(1, visualize_every=visualize_every)
            )
            test_errors.append(
                self.test_n_observations(2, visualize_every=visualize_every)
            )
            test_errors.append(
                self.test_n_observations(3, visualize_every=visualize_every)
            )
            test_errors.append(
                self.test_n_observations(4, visualize_every=visualize_every)
            )
            # self.test_n_observations(5, visualize_every=visualize_every)

        with open(f"test_errors_{self._run_name}.pickle", "wb") as f:
            compressed_pkl = blosc.compress_pickle(pickle.dumps(test_errors))
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
        auc = torch.trapz(
            torch.tensor([x["[PRIOR] MPJPE (mm)"] for x in test_errors]),
            torch.tensor([1, 2, 3, 4]),
        )
        plt.title(f"Test MPJPE and Root-aligned MPJPE for N observations")
        # Add the AUC to the legend:
        plt.legend(title=f"MPJPE AUC: {auc:.2f}")
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
        auc = torch.trapz(
            torch.tensor([x["[PRIOR] MPJPE (mm)"] for x in test_errors]),
            torch.tensor([1, 2, 3, 4]),
        )
        plt.title(f"Test MPJPE for N observations")
        # Add the AUC to the legend:
        plt.legend(title=f"MPJPE AUC: {auc:.2f}")
        plt.xlabel("N observations")
        plt.ylabel("MPJPE (mm)")
        plt.savefig(f"test_error_mpjpe_only_{self._run_name}.png")
        plt.show()
