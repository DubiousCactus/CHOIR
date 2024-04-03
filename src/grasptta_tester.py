#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2024 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Test GraspTTA (GraspCVAE + ContactNet + TTA) on a dataset.
"""

from typing import Dict, Optional

import torch

from model.affine_mano import AffineMANO
from src.multiview_tester import MultiViewTester
from utils import to_cuda_


class GraspTTATester(MultiViewTester):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._data_loader.dataset.set_observations_number(1)
        self.affine_mano: AffineMANO = to_cuda_(AffineMANO(for_contactpose=True))  # type: ignore

    def _inference(
        self,
        samples,
        labels,
        max_observations: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        max_observations = max_observations or samples["choir"].shape[1]
        gt_hand_params = torch.cat(
            (labels["theta"], labels["beta"], labels["trans"], labels["rot"]), dim=-1
        )[:, -1]

        gt_verts, _ = self.affine_mano(
            labels["theta"][:, -1],
            labels["beta"][:, -1],
            labels["trans"][:, -1],
            rot_6d=labels["rot"][:, -1],
        )

        recon_xyz, recon_joints, recon_anchors, recon_param = self._model(
            labels["obj_pts"][:, -1],
            gt_verts,
        )
        return {
            "verts": recon_xyz,
            "joints": recon_joints,
            "anchors": recon_anchors,
            "faces": self.affine_mano.faces,
        }
