#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Hand-Object Interaction loss.
"""

from typing import Dict, Tuple

import torch
from torch.distributions import kl_divergence

from model.affine_mano import AffineMANO


class CHOIRLoss(torch.nn.Module):
    def __init__(
        self,
        bps: torch.Tensor,
        predict_anchor_orientation: bool,
        predict_anchor_position: bool,
        predict_mano: bool,
        orientation_w: float,
        distance_w: float,
        assignment_w: float,
        mano_pose_w: float,
        mano_global_pose_w: float,
        mano_shape_w: float,
        mano_agreement_w: float,
        mano_anchors_w: float,
        kl_w: float,
        multi_view: bool,
    ) -> None:
        super().__init__()
        self._mse = torch.nn.MSELoss()
        self._cosine_embedding = torch.nn.CosineEmbeddingLoss()
        self._cross_entropy = torch.nn.CrossEntropyLoss()
        self._predict_anchor_orientation = predict_anchor_orientation
        self._predict_anchor_position = predict_anchor_position
        self._predict_mano = predict_mano
        self._orientation_w = orientation_w
        self._distance_w = distance_w
        self._assignment_w = assignment_w
        self._mano_pose_w = mano_pose_w
        self._mano_global_pose_w = mano_global_pose_w
        self._mano_shape_w = mano_shape_w
        self._mano_agreement_w = mano_agreement_w
        self._mano_anchors_w = mano_anchors_w
        self._kl_w = kl_w
        self._affine_mano = AffineMANO()
        self._hoi_loss = CHOIRFittingLoss() if predict_mano else None
        self._multi_view = multi_view
        self.register_buffer("bps", bps)

    def forward(
        self,
        samples: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor],
        y_hat,
    ) -> torch.Tensor:
        scalar = samples["scalar"]
        (
            choir_gt,
            scalar_gt,
            joints_gt,
            anchors_gt,
            pose_gt,
            beta_gt,
            rot_gt,
            trans_gt,
        ) = (
            labels["choir"],
            labels["scalar"],
            labels["joints"],
            labels["anchors"],
            labels["theta"],
            labels["beta"],
            labels["rot"],
            labels["trans"],
        )
        if self._multi_view:
            for n in range(choir_gt.shape[1] - 1):
                assert torch.allclose(choir_gt[:, n, 0], choir_gt[:, n + 1, 0])
            # Since all noisy views are of the same grasp, we can just take the first view's ground
            # truth (all views have the same ground truth).
            choir_gt = choir_gt[:, 0]
            pose_gt = pose_gt[:, 0]
            beta_gt = beta_gt[:, 0]
            rot_gt = rot_gt[:, 0]
            trans_gt = trans_gt[:, 0]
            anchors_gt = anchors_gt[:, 0]

        choir_pred, orientations_pred = y_hat["choir"], y_hat["orientations"]
        # choir_gt, orientations_gt = choir_gt, anchor_orientations
        losses = {
            "distances": self._distance_w
            * self._mse(choir_gt[:, :, 1], choir_pred[:, :, 1])
        }
        anchor_positions_pred = None
        if y_hat["posterior"] is not None and y_hat["prior"] is not None:
            losses["kl_div"] = (
                kl_divergence(y_hat["posterior"], y_hat["prior"]).mean() * self._kl_w
            )
        if self._predict_anchor_orientation or self._predict_anchor_position:
            raise NotImplementedError
            B, P, D = orientations_pred.shape
            losses["orientations"] = self._orientation_w * self._cosine_embedding(
                orientations_pred.reshape(B * P, D),
                orientations_gt.reshape(B * P, D),
                torch.ones(B * P, device=orientations_pred.device),
            )
            if self._predict_anchor_position:
                target_anchor_positions = orientations_gt * choir_gt[:, :, 4].unsqueeze(
                    -1
                ).repeat(1, 1, 3)
                anchor_positions_pred = orientations_pred * choir_pred[
                    :, :, 4
                ].unsqueeze(-1).repeat(1, 1, 3)
                losses["positions"] = self._distance_w * self._mse(
                    target_anchor_positions, anchor_positions_pred
                )
        if self._predict_mano:
            mano = y_hat["mano"]
            pose, shape, rot, trans = (
                mano[:, :18],
                mano[:, 18 : 18 + 10],
                mano[:, 18 + 10 : 18 + 10 + 6],
                mano[:, 18 + 10 + 6 :],
            )
            # Penalize high shape values to avoid exploding shapes and unnatural hands:
            shape_reg = torch.norm(shape, dim=-1).mean()  # ** 2
            # The MANO parameters and resulting anchor vertices are in the BPS coordinate system
            # and scale, such that the agreement loss is valid to the predicted CHOIR field.
            # Alternatively, we can rescale the CHOIR field to the original MANO coordinate system and
            # this way we can supervise the MANO parameters in the original coordinate system.
            verts, _ = self._affine_mano(pose, shape, rot, trans)
            anchors = self._affine_mano.get_anchors(verts)
            anchor_agreement_loss, _ = self._hoi_loss(
                anchors,
                choir_pred / scalar[:, None],
                self.bps.unsqueeze(0).repeat(choir_pred.shape[0], 1, 1)
                / scalar[:, None],
            )
            losses["mano_pose"] = self._mano_pose_w * self._mse(pose, pose_gt)
            losses["mano_shape"] = self._mano_shape_w * (
                shape_reg + self._mse(shape, beta_gt)
            )
            losses["mano_global_pose"] = self._mano_global_pose_w * (
                self._mse(rot, rot_gt) + self._mse(trans, trans_gt)
            )
            losses["mano_anchors"] = self._mano_anchors_w * self._mse(
                anchors, anchors_gt
            )
            losses["mano_anchor_agreement"] = (
                self._mano_agreement_w * anchor_agreement_loss
            )
            # TODO: Penalize MANO penetration into the object pointcloud (recoverable through the
            # input CHOIR field which must include the delta vectors, or we can pass the delta
            # vectors to this loss separately).
        return losses


class CHOIRFittingLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        # verts: torch.Tensor,
        anchors: torch.Tensor,
        choir: torch.Tensor,
        bps: torch.Tensor,
        # bps_mean: torch.Tensor,
        # bps_scalar: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(bps.shape) == 2:
            assert bps.shape[0] % 32 == 0, "bps_dim must be a multiple of 32"
        elif len(bps.shape) == 3:
            assert bps.shape[1] % 32 == 0, "bps_dim must be a multiple of 32"
        """
        Args:
            anchors (torch.Tensor): (B, 32, 3) The predicted MANO anchors in the BPS coordinate system.
            choir (torch.Tensor): (B, P, 2) The predicted CHOIR field in the BPS coordinate system.
                                            P is the number of points in the BPS representation and
                                            the last dimension is composed of: (a) the nearest
                                            object BPS distance, (b) the MANO anchor BPS distance
                                            (not nearest but fixed anchors).
            bps (torch.Tensor): (B, P, 3) The BPS representation of the object point cloud. Since
                                            it is fixed, it would make sense to be (P, 3) but we
                                            rescale the BPS according to the scale of the
                                            hand-object pair associated with batch element n, so we
                                            duplicate the BPS for each batch element and apply the
                                            corresponding scalar.

        This loss should be computed as MSE between the computed distance
        from the predicted anchors to the reconstructed target point cloud and the encoded
        distances in the choir field.
        IMPORTANT: The anchors and the CHOIR field must be in the same coordinate frame and scale.
        When doing test-time optimization, we pass the rescaled BPS so that we can predict MANO in
        the camera frame and not in the original BPS frame.
        """
        # Step 1: Compute the distance from the predicted anchors to the reconstructed target point cloud
        # tgt_points = self.bps.decode(x_deltas=choir[:, :, 1:4])
        # tgt_points = denormalize(tgt_points, bps_mean, bps_scalar)
        if len(bps.shape) == 2:
            assert bps.shape[0] == choir.shape[1]
        elif len(bps.shape) == 3:
            assert bps.shape[1] == choir.shape[1]
        else:
            raise ValueError("CHOIRFittingLoss(): BPS must be (B, P, 3) or (P, 3)")
        anchor_distances = torch.cdist(bps, anchors, p=2)
        anchor_ids = (
            torch.arange(
                0,
                anchors.shape[1],
                device=choir.device,
            )
            .repeat((choir.shape[1] // 32,))
            .unsqueeze(0)
            .repeat((choir.shape[0], 1))
            .unsqueeze(-1)
        )
        distances = torch.gather(anchor_distances, 2, anchor_ids).squeeze(-1)
        choir_loss = torch.nn.functional.mse_loss(distances, choir[:, :, -1])
        return choir_loss
