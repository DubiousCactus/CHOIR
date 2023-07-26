#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.


from typing import Dict, List, Tuple, Union

import torch

from src.base_trainer import BaseTrainer
from utils import to_cuda
from utils.training import get_dict_from_sample_and_label_tensors
from utils.visualization import visualize_model_predictions_with_multiple_views


class MultiViewTrainer(BaseTrainer):
    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
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
        visualize_model_predictions_with_multiple_views(
            self._model,
            batch,
            epoch,
            bps_dim=self._bps_dim,
            bps=self._bps,
            remap_bps_distances=self._remap_bps_distances,
            exponential_map_w=self._exponential_map_w,
        )  # User implementation goes here (utils/visualization.py)

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
        x, y = batch
        samples, labels = get_dict_from_sample_and_label_tensors(x, y)
        y_hat = self._model(samples["choir"], labels["choir"])
        # If we're in validation mode, let's rescale the CHOIR prediction and ground-truth so that
        # all metrics are comparable  between different scaling modes, etc.
        losses = self._training_loss(
            samples, labels, y_hat, rescale=validation
        )  # TODO: Multiview + Contrastive Learning loss
        loss = sum([v for v in losses.values()])
        # Again without using the posterior:
        y_hat = self._model(samples["choir"])
        losses["distances_from_prior"] = self._training_loss(
            samples, labels, y_hat, rescale=validation
        )["distances"]
        return loss, losses
