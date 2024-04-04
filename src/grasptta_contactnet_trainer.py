#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2024 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.


"""
ContactNet trainer for GraspTTA.
"""

from typing import Dict, List, Tuple, Union

import torch

from model.affine_mano import AffineMANO
from src.base_trainer import BaseTrainer
from utils import to_cuda, to_cuda_


class ContactNetTrainer(BaseTrainer):
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
        raise NotImplementedError("Implement the visualization method.")

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
        gt_verts, _ = self.affine_mano(
            labels["theta"][:, -1],
            labels["beta"][:, -1],
            labels["trans"][:, -1],
            rot_6d=labels["rot"][:, -1],
        )

        obj_pts = labels["obj_pts"][:, -1].permute(0, 2, 1)
        obj_contacts = labels["obj_contacts"][:, -1].squeeze(dim=-1)

        recon_cmap = self._model(obj_pts, gt_verts.permute(0, 2, 1))
        # but why not use MSE mean reduction directly???? I'm just copying the code from the original implementation.
        loss = torch.nn.functional.mse_loss(
            recon_cmap, obj_contacts, reduction="none"
        ).sum() / recon_cmap.size(0)
        return loss, {}
