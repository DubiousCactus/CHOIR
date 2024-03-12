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
from utils.dataset import fetch_gaussian_params_from_CHOIR
from utils.visualization import visualize_model_predictions_with_multiple_views


class MultiViewDDPMBaselineTrainer(MultiViewDDPMTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._train_loader.dataset.set_eval_mode(True)
        self._val_loader.dataset.set_eval_mode(True)

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
            method="baseline",
        )  # User implementation goes here (utils/training.py)

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
        x = (
            torch.cat(
                (
                    labels["rescaled_ref_pts"][:, -1],
                    labels["joints"][:, -1],
                    labels["anchors"][:, -1],
                ),
                dim=-2,  # Concat along the keypoints and not their dimensionality
            )
            if self._full_choir  # full_hand_object_pair
            else torch.cat((labels["joints"][:, -1], labels["anchors"][:, -1]), dim=-2)
        )
        y_modality = self._sample_modality(epoch)
        if y_modality == "object":
            y = samples["rescaled_ref_pts"] if self.conditional else None
        else:
            y = (
                torch.cat(
                    (
                        samples["rescaled_ref_pts"],
                        samples["joints"],
                        samples["anchors"],
                    ),
                    dim=-2,
                )
                if self.conditional
                else None
            )
        kwargs = {"x": x, "y": y, "y_modality": y_modality}
        if self._model_contacts:
            kwargs["contacts"] = fetch_gaussian_params_from_CHOIR(
                labels["choir"].squeeze(1),
                self._train_loader.dataset.anchor_indices,
                n_repeats=self._train_loader.dataset.bps_dim // 32,
                n_anchors=32,
                choir_includes_obj=True,
            )[:, :, 0].squeeze(1)

        y_hat = self._model(**kwargs)
        losses = self._training_loss(None, None, y_hat)
        loss = sum([v for v in losses.values()])
        losses["contacts_mse"] /= self._training_loss.contacts_weight
        return loss, losses
