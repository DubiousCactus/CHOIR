#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.


from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.base_trainer import BaseTrainer
from utils import to_cuda
from utils.training import get_dict_from_sample_and_label_tensors
from utils.visualization import visualize_model_predictions_with_multiple_views


class MultiViewTrainer(BaseTrainer):
    def __init__(
        self,
        run_name: str,
        model: torch.nn.Module,
        opt: Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        training_loss: torch.nn.Module,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            run_name,
            model,
            opt,
            train_loader,
            val_loader,
            training_loss,
            scheduler,
        )
        self._fine_tune = kwargs.get("fine_tune", False)

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
        x, y, _ = batch
        samples, labels = get_dict_from_sample_and_label_tensors(x, y)
        # If we're fine tuning, we'll skip the labels and train the prior!
        y_hat = self._model(
            samples["choir"], labels["choir"] if not self._fine_tune else None
        )
        # If we're in validation mode, let's rescale the CHOIR prediction and ground-truth so that
        # all metrics are comparable  between different scaling modes, etc.
        losses = self._training_loss(
            samples, labels, y_hat, rescale=validation
        )  # TODO: Contrastive Learning loss
        loss = sum([v for v in losses.values()])
        # Again without using the posterior:
        y_hat = self._model(samples["choir"])
        losses["distances_from_prior"] = self._training_loss(
            samples, labels, y_hat, rescale=validation
        )["distances"]
        return loss, losses
