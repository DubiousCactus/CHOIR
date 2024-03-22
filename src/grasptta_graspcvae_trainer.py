#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2024 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
GraspCVAE trainer for GraspTTA.
"""

from typing import Dict, List, Tuple, Union

import torch

from src.base_trainer import BaseTrainer
from utils import to_cuda


class GraspCVAETrainer(BaseTrainer):
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
        samples, labels, _ = batch
        gt_hand_params = torch.cat(
            (labels["theta"], labels["beta"], labels["rot"], labels["trans"]), dim=-1
        )
        recon_param, mean, log_var, z = self.model(labels["obj_pts"], gt_hand_params)
        loss = torch.tensor(0.0)
        losses = {}
        raise NotImplementedError("Implement the training/validation iteration method.")
        return loss, losses
