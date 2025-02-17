#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2024 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
GraspCVAE trainer for GraspTTA.
"""

from typing import Dict, List, Tuple, Union

import torch
import trimesh

from model.affine_mano import AffineMANO
from src.base_trainer import BaseTrainer
from utils import to_cuda, to_cuda_
from utils.visualization import visualize_MANO


class GraspCVAETrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO: For now it's only for contactpose, later we may use OakInk as well.
        self.affine_mano: AffineMANO = to_cuda_(AffineMANO(for_contactpose=True))  # type: ignore

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
        samples, labels, _ = batch
        gt_verts, _ = self.affine_mano(
            labels["theta"][:, -1],
            labels["beta"][:, -1],
            labels["trans"][:, -1],
            rot_6d=labels["rot"][:, -1],
        )
        gt_hand_mesh = trimesh.Trimesh(
            vertices=gt_verts[0].detach().cpu().numpy(),
            faces=self.affine_mano.faces.detach().cpu().numpy(),
        )
        obj_ptcld = labels["obj_pts"][:, -1].cpu().numpy()

        recon_param, mean, log_var, z = self._model(
            labels["obj_pts"][0, -1].permute(0, 2, 1), gt_verts.permute(0, 2, 1)
        )
        recon_verts, _ = self.affine_mano(
            recon_param[:, :18],
            recon_param[:, 18:28],
            recon_param[:, 28:31],
            rot_6d=recon_param[:, 31:37],
        )
        pred_hand_mesh = trimesh.Trimesh(
            vertices=recon_verts[0].detach().cpu().numpy(),
            faces=self.affine_mano.faces.detach().cpu().numpy(),
        )
        visualize_MANO(
            pred_hand_mesh, obj_ptcld=obj_ptcld, gt_hand=gt_hand_mesh, opacity=1.0
        )

    @to_cuda
    def _train_val_iteration(
        self,
        batch: Union[Tuple, List, torch.Tensor],
        epoch: int,
        validation: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Training or validation procedure for one batch. We want to keep the code DRY and avoid
        making mistakes, so write this code only once at the cost of many function calls!
        Args:
            batch: The batch to process.
        Returns:
            torch.Tensor: The loss for the batch.
        """
        _, labels, _ = batch
        gt_hand_params = torch.cat(
            (labels["theta"], labels["beta"], labels["trans"], labels["rot"]), dim=-1
        )[:, -1]

        gt_verts, _ = self.affine_mano(
            labels["theta"][:, -1],
            labels["beta"][:, -1],
            labels["trans"][:, -1],
            rot_6d=labels["rot"][:, -1],
        )

        recon_param, mean, log_var, z = self._model(
            labels["obj_pts"][:, -1].permute(0, 2, 1), gt_verts.permute(0, 2, 1)
        )

        # TODO: Implement for OakInk
        recon_verts, _ = self.affine_mano(
            recon_param[:, :18],
            recon_param[:, 18:28],
            recon_param[:, 28:31],
            rot_6d=recon_param[:, 31:37],
        )

        loss, loss_items = self._training_loss(
            recon_param,
            gt_hand_params,
            mean,
            log_var,
            recon_verts,
            gt_verts,
            self.affine_mano.faces.detach().unsqueeze(0).repeat(gt_verts.size(0), 1, 1),
            labels["obj_pts"][:, -1],
            epoch,
            training_mode=not validation,
        )
        return loss, loss_items
