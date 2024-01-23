#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2024 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.


from typing import Dict, List, Tuple, Union

import torch

from src.base_trainer import BaseTrainer
from utils import to_cuda
from utils.visualization import visualize_model_predictions_with_multiple_views


class MultiViewDDPMTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conditional = kwargs.get("conditional", False)
        self._use_deltas = self._train_loader.dataset.use_deltas
        self._full_choir = kwargs.get("full_choir", False)

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

        if not self._use_deltas:
            y_hat = self._model(
                # Take the last frame
                labels["choir"][:, -1]
                if self._full_choir
                else labels["choir"][:, -1][..., -1].unsqueeze(-1),
                samples["choir"] if self.conditional else None,
            )  # Only the hand distances!
        else:
            raise NotImplementedError(
                "Have to implement it with embed_full_choir in DiffusionModel!"
            )
            y_hat = self._model(
                labels["choir"][:, -1]
                if self._full_choir
                else labels["choir"][:, -1][..., 3:],
                samples["choir"] if self.conditional else None,
            )  # Only the hand deltas!
        losses = self._training_loss(
            samples, {k: v[:, -1] for k, v in labels.items()}, y_hat
        )
        loss = sum([v for v in losses.values()])
        return loss, losses
