#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2024 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.


from typing import Dict, List, Tuple, Union

import torch

from src.multiview_ddpm_trainer import MultiViewDDPMTrainer
from utils import to_cuda
from utils.visualization import visualize_model_predictions_with_multiple_views


class MultiViewDDPMBaselineTrainer(MultiViewDDPMTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
        visualize_model_predictions_with_multiple_views(
            self._model,
            (samples, labels, None),
            epoch,
            bps_dim=self._bps_dim,
            bps=self._bps,
            anchor_indices=self._anchor_indices,
            remap_bps_distances=self._remap_bps_distances,
            exponential_map_w=self._exponential_map_w,
            dataset=self._train_loader.dataset.name,
            theta_dim=self._train_loader.dataset.theta_dim,
            use_deltas=self._use_deltas,
            conditional=self.conditional,
            method="ddpm",
        )  # User implementation goes here (utils/training.py)

    @to_cuda
    def _train_val_iteration(
        self,
        batch: Union[Tuple, List, torch.Tensor],
        validation: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Training or validation procedure for one batch. We want to keep the code DRY and avoid
        making mistakes, so write this code only once at the cost of many function calls!
        Args:
            batch: The batch to process.
        Returns:
            torch.Tensor: The loss for the batch.
        """
        samples, labels, _ = batch
        y_hat = self._model(
            # Take the last frame
            torch.cat(
                (
                    labels["rescaled_ref_pts"][:, -1],
                    labels["joints"][:, -1],
                    labels["anchors"][:, -1],
                ),
                dim=-2,  # Concat along the keypoints and not their dimensionality
            )
            if self._full_choir  # full_hand_object_pair
            else torch.cat((labels["joints"][:, -1], labels["anchors"][:, -1]), dim=-2),
            torch.cat(
                (samples["rescaled_ref_pts"], samples["joints"], samples["anchors"]),
                dim=-2,
            )
            if self.conditional
            else None,
        )  # Only the hand distances!
        losses = self._training_loss(
            samples, {k: v[:, -1] for k, v in labels.items()}, y_hat
        )
        loss = sum([v for v in losses.values()])
        return loss, losses
