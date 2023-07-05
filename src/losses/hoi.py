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
    def __init__(self) -> None:
        super().__init__()
        self._mse = torch.nn.MSELoss()
        self._cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(
        self,
        y,
        y_hat,
    ) -> torch.Tensor:
        loss = {
            "distances": self._mse(y[:, :, :-1], y_hat[:, :, :-1]),
            "anchors": self._cross_entropy(y[:, :, -1], y_hat[:, :, -1]),
        }
        return loss


class DualHOILoss(torch.nn.Module):
    def __init__(self, bps_dim: int):
        super().__init__()
        self.bps = bps_torch(
            bps_type="random_uniform",
            n_bps_points=bps_dim,
            radius=1.0,
            n_dims=3,
            custom_basis=None,
        )

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
        # Step 2: Mask the distances by the anchor indices in the choir field
        anchor_distances = anchor_distances.gather(
            dim=2, index=choir[:, :, -1].long().unsqueeze(dim=2)
        ).squeeze(dim=2)
        # Step 3: Compute the MSE between the masked distances and the encoded distances in the choir field
        choir_loss = torch.nn.functional.mse_loss(anchor_distances, choir[:, :, -2])
        # Now add the hand contact loss
        hand_contact_loss = (
            torch.nn.functional.mse_loss(
                hand_contacts, compute_hand_contacts_simple(tgt_points, verts)
            )
            if hand_contacts is not None
            else torch.tensor(0.0)
        )
        return choir_loss, hand_contact_loss
