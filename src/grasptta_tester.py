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
