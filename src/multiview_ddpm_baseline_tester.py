#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2024 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.


from typing import Dict, List, Optional, Tuple, Union

import torch

from src.multiview_tester import MultiViewTester
from utils import to_cuda, to_cuda_
from utils.visualization import visualize_model_predictions_with_multiple_views


class MultiViewDDPMBaselineTester(MultiViewTester):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conditional = kwargs.get("conditional", False)
        self._data_loader.dataset.set_observations_number(1)
        self._use_deltas = self._data_loader.dataset.use_deltas
        self._full_choir = kwargs.get("full_choir", False)
        self._is_baseline = True
        # Because I infer the shape of the model from the data, I need to
        # run the model's forward pass once before calling .generate()
        print("[*] Running the model's forward pass once...")
        with torch.no_grad():
            samples, labels, _ = to_cuda_(next(iter(self._data_loader)))
            self._model(
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
                else torch.cat(
                    (labels["joints"][:, -1], labels["anchors"][:, -1]), dim=-2
                ),
                torch.cat(
                    (
                        samples["rescaled_ref_pts"],
                        samples["joints"],
                        samples["anchors"],
                    ),
                    dim=-2,
                )
                if self.conditional
                else None,
            )
        print("[+] Done!")

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
            anchor_indices=self._anchor_indices,
            remap_bps_distances=self._remap_bps_distances,
            exponential_map_w=self._exponential_map_w,
            dataset=self._data_loader.dataset.name,
            theta_dim=self._data_loader.dataset.theta_dim,
            use_deltas=self._use_deltas,
            conditional=self.conditional,
            method="baseline",
        )  # User implementation goes here (utils/training.py)

    # @to_cuda
    def _inference(
        self,
        samples,
        labels,
        max_observations: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Training or validation procedure for one batch. We want to keep the code DRY and avoid
        making mistakes, so write this code only once at the cost of many function calls!
        Args:
            batch: The batch to process.
        Returns:
            torch.Tensor: The loss for the batch.
        """
        max_observations = max_observations or samples["choir"].shape[1]
        pred = self._model.generate(
            1,
            y=torch.cat(
                (
                    samples["rescaled_ref_pts"][:, :max_observations],
                    samples["joints"][:, :max_observations],
                    samples["anchors"][:, :max_observations],
                ),
                dim=-2,
            )
            if self.conditional
            else None,
        ).squeeze(
            1
        )  # Only use 1 sample for now. TODO: use more samples and average?
        return {"hand_keypoints": pred}
