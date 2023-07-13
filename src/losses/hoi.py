#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Hand-Object Interaction loss.
"""

from typing import Optional, Tuple

import torch
from bps_torch.bps import bps_torch
from bps_torch.tools import denormalize

from utils.dataset import compute_hand_contacts_simple


class CHOIRLoss(torch.nn.Module):
    def __init__(
        self,
        anchor_assignment: str,
        predict_anchor_orientation: bool,
        predict_mano: bool,
    ) -> None:
        super().__init__()
        self._mse = torch.nn.MSELoss()
        self._cosine_similarity = torch.nn.CosineSimilarity(dim=2)
        self._cross_entropy = torch.nn.CrossEntropyLoss()
        self._anchor_assignment = anchor_assignment
        self._predict_anchor_orientation = predict_anchor_orientation
        self._predict_mano = predict_mano

    def forward(
        self,
        y,
        y_hat,
    ) -> torch.Tensor:
        if self._anchor_assignment not in ["random", "closest", "batched_fixed"]:
            raise NotImplementedError(
                f"Anchor assignment {self._anchor_assignment} not implemented."
            )
        (
            choir_gt,
            anchor_orientations,
            joints_gt,
            anchors_gt,
            pose_gt,
            beta_gt,
            rot_gt,
            trans_gt,
        ) = y
        choir_pred, orientations_pred = y_hat["choir"], y_hat["orientations"]
        choir_gt, target_anchor_orientations = choir_gt, anchor_orientations
        loss = {
            "distances": self._mse(choir_gt[:, :, 4], choir_pred[:, :, 4]) * 1000,
        }
        if self._predict_anchor_orientation:
            loss["orientations"] = self._cosine_similarity(
                orientations_pred, target_anchor_orientations
            )
        if self._anchor_assignment == "closest":
            loss["anchors"] = self._cross_entropy(y[:, :, -32:], y_hat[:, :, -32:])
        return loss


class DualHOILoss(torch.nn.Module):
    def __init__(self, bps_dim: int, anchor_assignment: str):
        super().__init__()
        if anchor_assignment == "batched_fixed":
            assert (
                bps_dim % 32 == 0
            ), "bps_dim must be a multiple of 32 for batched_fixed anchor assignment"
        self.bps = bps_torch(
            bps_type="random_uniform",
            n_bps_points=bps_dim,
            radius=1.0,
            n_dims=3,
            custom_basis=None,
        )
        self._anchor_assignment = anchor_assignment

    def forward(
        self,
        verts: torch.Tensor,
        anchors: torch.Tensor,
        choir: torch.Tensor,
        hand_contacts: Optional[torch.Tensor],
        bps_mean: torch.Tensor,
        bps_scalar: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The choir field has shape (B, P, 3) where P is the number of points in the BPS
        representation and the dimension is composed of: (a) the BPS distance, (b) the
        distance to the associated MANO anchor from the target point, and (c) the index
        of the associated MANO anchor.
        The anchors are specific MANO vertices and have shape (B, A, 3) where A is the number of
        anchors (typically 32). This loss should be computed as MSE between the computed distance
        from the predicted anchors to the reconstructed target point cloud and the encoded
        distances in the choir field.
        """
        # Step 1: Compute the distance from the predicted anchors to the reconstructed target point cloud
        tgt_points = self.bps.decode(x_deltas=choir[:, :, 1:4])
        tgt_points = denormalize(tgt_points, bps_mean, bps_scalar)
        anchor_distances = torch.cdist(tgt_points, anchors, p=2)

        if self._anchor_assignment in [
            "closest",
            "random",
        ]:
            choir_one_hot = choir[:, :, -32:]
            index = torch.argmax(choir_one_hot, dim=-1)
            assigned_anchor_distances = anchor_distances.gather(
                dim=2, index=index.unsqueeze(dim=2)
            ).squeeze(dim=2)
            # Step 3: Compute the MSE between the masked distances and the encoded distances in the choir field
            choir_loss = torch.nn.functional.mse_loss(
                assigned_anchor_distances, choir[:, :, -33]
            )
        elif self._anchor_assignment == "closest_and_farthest":
            closest_one_hot = choir[:, :, 5:37]
            closest_index = torch.argmax(closest_one_hot, dim=-1)
            lowest_anchor_distances = anchor_distances.gather(
                dim=2, index=closest_index.unsqueeze(dim=2)
            ).squeeze(dim=2)
            farthest_one_hot = choir[:, :, -32:]
            farthest_index = torch.argmax(farthest_one_hot, dim=-1)
            highest_anchor_distances = anchor_distances.gather(
                dim=2, index=farthest_index.unsqueeze(dim=2)
            ).squeeze(dim=2)
            choir_loss = torch.nn.functional.mse_loss(
                lowest_anchor_distances, choir[:, :, 4]
            ) + torch.nn.functional.mse_loss(highest_anchor_distances, choir[:, :, -33])
        elif self._anchor_assignment == "batched_fixed":
            anchor_ids = (
                torch.arange(
                    0,
                    anchors.shape[1],
                    device=choir.device,
                )
                .repeat((choir.shape[1] // 32,))
                .unsqueeze(0)
            )
            distances = torch.gather(
                anchor_distances, 2, anchor_ids.unsqueeze(-1)
            ).squeeze(-1)
            choir_loss = torch.nn.functional.mse_loss(distances, choir[:, :, -1])
        else:
            raise ValueError(
                f"Unknown anchor assignment method: {self._anchor_assignment}"
            )
        # Now add the hand contact loss
        hand_contact_loss = (
            torch.nn.functional.mse_loss(
                hand_contacts, compute_hand_contacts_simple(tgt_points, verts)
            )
            if hand_contacts is not None
            else torch.tensor(0.0)
        )
        return choir_loss, hand_contact_loss
