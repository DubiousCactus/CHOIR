#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2024 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Test GraspTTA (GraspCVAE + ContactNet + TTA) on a dataset.
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import trimesh

from model.affine_mano import AffineMANO
from src.multiview_tester import MultiViewTester
from utils import to_cuda, to_cuda_
from utils.visualization import visualize_MANO


class GraspTTATester(MultiViewTester):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._data_loader.dataset.set_observations_number(1)
        self.affine_mano: AffineMANO = to_cuda_(AffineMANO(for_contactpose=True))  # type: ignore
        self._is_grasptta = True

    def _inference(
        self,
        samples,
        labels,
        max_observations: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        recon_xyz, recon_joints, recon_anchors, _ = self._model(
            labels["obj_pts"][:, -1],
        )
        return {
            "verts": recon_xyz,
            "joints": recon_joints,
            "anchors": recon_anchors,
            "faces": self.affine_mano.faces,
        }

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
