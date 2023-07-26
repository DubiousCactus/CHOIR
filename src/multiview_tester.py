#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.


import signal
from collections import defaultdict
from typing import Dict, List, Tuple, Union

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
        self._pbar = tqdm(total=len(self._data_loader), desc="Testing")
        self._affine_mano = to_cuda_(AffineMANO())
        self._bps_dim = data_loader.dataset.bps_dim
        self._bps = to_cuda_(data_loader.dataset.bps)
        self._remap_bps_distances = data_loader.dataset.remap_bps_distances
        self._exponential_map_w = data_loader.dataset.exponential_map_w
        signal.signal(signal.SIGINT, self._terminator)

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
        x, y = batch  # type: ignore
        samples, labels = get_dict_from_sample_and_label_tensors(x, y)
        input_scalar = samples["scalar"]
        if len(input_scalar.shape) == 2:
            input_scalar = input_scalar.mean(
                dim=1
            )  # TODO: Think of a better way for 'pair' scaling
        y_hat = self._model(samples["choir"])
        mano_params_gt = {
            "pose": labels["theta"],
            "beta": labels["beta"],
            "rot_6d": labels["rot"],
            "trans": labels["trans"],
        }
        # Only use the first view for each batch element
        mano_params_gt = {k: v[:, 0] for k, v in mano_params_gt.items()}
        gt_pose, gt_shape, gt_rot_6d, gt_trans = tuple(mano_params_gt.values())
        gt_verts, gt_joints = self._affine_mano(gt_pose, gt_shape, gt_rot_6d, gt_trans)
        gt_anchors = self._affine_mano.get_anchors(gt_verts)
        with torch.set_grad_enabled(True):
            pose, shape, rot_6d, trans, anchors_pred = optimize_pose_pca_from_choir(
                y_hat["choir"],
                bps=self._bps,
                scalar=input_scalar,
                max_iterations=5000,
                loss_thresh=1e-14,
                lr=3e-2,
                remap_bps_distances=self._remap_bps_distances,
                exponential_map_w=self._exponential_map_w,
            )
            verts_pred, joints_pred = self._affine_mano(pose, shape, rot_6d, trans)
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
        return {
            "Anchor error (mm)": anchor_error,
            "MPJPE (mm)": mpjpe,
            "Root-aligned MPJPE (mm)": root_aligned_mpjpe,
            "MPVPE (mm)": mpvpe,
        }

    def test(self, visualize_every: int = 0):
        """Computes the average loss on the test set.
        Args:
            visualize_every (int, optional): Visualize the model predictions every n batches.
            Defaults to 0 (no visualization).
        """
        metrics = defaultdict(MeanMetric)
        self._pbar.reset()
        self._pbar.set_description("Testing")
        color_code = project_conf.ANSI_COLORS[project_conf.Theme.TESTING.value]
        self._model.eval()
        " ==================== Training loop for one epoch ==================== "
        with torch.no_grad():
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
        print("==" + " " * 31 + " Test results " + " " * 31 + "==")
        print("=" * 81)
        for k, v in metrics.items():
            print(f"\t -> {k}: {v.compute().item():.2f}")
        print("_" * 81)
