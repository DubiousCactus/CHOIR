#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
MANO with an affine transformation on its vertices.
"""

from typing import Tuple

import torch
from manotorch.anchorlayer import AnchorLayer
from manotorch.manolayer import ManoLayer, MANOOutput

from utils.dataset import transform_verts


class AffineMANO(torch.nn.Module):
    def __init__(
        self,
        ncomps: int = 15,
        flat_hand_mean: bool = False,
        for_contactpose: bool = False,
    ):
        super().__init__()
        # MANO is shipped with 15 components but you can use less.
        self.mano_layer = ManoLayer(
            mano_assets_root="vendor/manotorch/assets/mano",
            use_pca=True,
            flat_hand_mean=flat_hand_mean,
            ncomps=ncomps,
        )
        self.anchor_layer = AnchorLayer(anchor_root="vendor/manotorch/assets/anchor")
        self._for_contactpose = for_contactpose

    def forward(self, pose, shape, rot_6d, trans) -> Tuple[torch.Tensor, torch.Tensor]:
        mano_output: MANOOutput = self.mano_layer(pose, shape)
        verts = mano_output.verts
        joints = mano_output.joints
        return transform_verts(
            verts, rot_6d, trans, apply_inverse_rot=self._for_contactpose
        ), transform_verts(
            joints, rot_6d, trans, apply_inverse_rot=self._for_contactpose
        )

    @property
    def faces(self) -> torch.Tensor:
        return self.mano_layer.th_faces  # type: ignore

    @property
    def closed_faces(self) -> torch.Tensor:
        return self.mano_layer.get_mano_closed_faces()  # type: ignore

    def get_anchors(self, verts: torch.Tensor) -> torch.Tensor:
        return self.anchor_layer(verts)
