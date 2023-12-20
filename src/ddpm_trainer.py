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
from utils.visualization import visualize_ddpm_generation


class DDPMTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conditional = kwargs.get("conditional", False)

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
        visualize_ddpm_generation(
            self._model,
            batch,
            epoch,
            bps_dim=self._bps_dim,
            bps=self._bps,
            anchor_indices=self._anchor_indices,
            remap_bps_distances=self._remap_bps_distances,
            exponential_map_w=self._exponential_map_w,
            dataset=self._train_loader.dataset.name,
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
        # For this baseline, we onl want one batch dimension so we can reshape all tensors to be (B * T, ...):
        for k, v in samples.items():
            # samples[k] = v.view(-1, *v.shape[2:])
            samples[k] = v[:, 0, ...]
        for k, v in labels.items():
            # labels[k] = v.view(-1, *v.shape[2:])
            labels[k] = v[:, 0, ...]
        y_hat = self._model(
            samples["choir"][..., -1].unsqueeze(-1),
            samples["choir"][..., 0].unsqueeze(-1) if self.conditional else None,
        )  # Only the hand distances!
        losses = self._training_loss(samples, labels, y_hat)
        loss = sum([v for v in losses.values()])
        return loss, losses
