#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2024 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.


from typing import Dict, Optional

import torch

from model.affine_mano import AffineMANO
from src.multiview_tester import MultiViewTester
from utils import to_cuda_


class GraspCVAETester(MultiViewTester):
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
        recon_param = self._model.inference(labels["obj_pts"][:, -1].permute(0, 2, 1))

        # TODO: Implement for OakInk
        recon_verts, recon_joints = self.affine_mano(
            recon_param[:, :18],
            recon_param[:, 18:28],
            recon_param[:, 28:31],
            rot_6d=recon_param[:, 31:37],
        )
        recon_anchors = self.affine_mano.get_anchors(recon_verts)
        return {
            "verts": recon_verts,
            "joints": recon_joints,
            "anchors": recon_anchors,
            "faces": self.affine_mano.faces,
        }
