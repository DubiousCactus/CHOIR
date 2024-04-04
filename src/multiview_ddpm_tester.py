#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2024 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.


import random
from typing import Dict, List, Optional, Tuple, Union

import torch

from src.multiview_tester import MultiViewTester
from utils import to_cuda, to_cuda_
from utils.visualization import visualize_model_predictions_with_multiple_views


class MultiViewDDPMTester(MultiViewTester):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conditional = kwargs.get("conditional", False)
        self._data_loader.dataset.set_observations_number(1)
        self._full_choir = kwargs.get("full_choir", False)
        self._use_deltas = self._data_loader.dataset.use_deltas
        self._model_contacts = kwargs.get("model_contacts", False)
        self._enable_contacts_tto = kwargs.get("enable_contacts_tto", False)
        self._use_ema = kwargs.get("use_ema", False)
        if self._model_contacts:
            self._model.backbone.set_anchor_indices(
                self._data_loader.dataset.anchor_indices
            )
            self._model.set_dataset_stats(self._data_loader.dataset)
            self._ema.ema_model.backbone.set_anchor_indices(
                self._data_loader.dataset.anchor_indices
            )
            self._ema.ema_model.set_dataset_stats(self._data_loader.dataset)
        self._single_modality = self._model.single_modality
        # Because I infer the shape of the model from the data, I need to
        # run the model's forward pass once before calling .generate()
        if kwargs.get("compile_test_model", False):
            print("[*] Compiling the model...")
            self._model = torch.compile(self._model)
            self._ema.ema_model = torch.compile(self._ema.ema_model)
        print("[*] Running the model's forward pass once...")
        with torch.no_grad():
            samples, labels, _ = to_cuda_(next(iter(self._data_loader)))
            x = (
                labels["choir"][:, -1]
                if self._full_choir
                else (
                    labels["choir"][:, -1][..., -1].unsqueeze(-1)
                    if not self._model_contacts
                    else (
                        labels["choir"][:, -1]
                        if self._model.object_in_encoder
                        else labels["choir"][:, -1][..., 1:]
                    )
                )
            )
            if self._single_modality is not None:
                modality = self._single_modality
            else:
                modality = random.choice(["noisy_pair", "object"])

            y = samples["choir"] if self.conditional else None

            if modality == "object":
                y = y[..., 0].unsqueeze(-1)
            elif modality == "noisy_pair":
                pass  # Already comes in noisy_pair modality
            self._model(x, y, y_modality=modality)
            self._ema.ema_model(x, y, y_modality=modality)
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
            method="coddpm",
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
        model = self._ema.ema_model if self._use_ema else self._model
        if self._single_modality is not None:
            modality = self._single_modality
        else:
            #modality = random.choice(["noisy_pair", "object"])
            modality = "object"
        y = samples["choir"][:, :max_observations] if self.conditional else None
        print(f"[*] Using modality: {modality}")
        if modality == "object":
            y = y[..., 0].unsqueeze(-1)
        elif modality == "noisy_pair":
            pass  # Already comes in noisy_pair modality
        udf, contacts = model.generate(
            1,
            y=y,
            y_modality=modality,
        )
        # Only use 1 sample for now. TODO: use more samples and average?
        udf, contacts = udf.squeeze(1), contacts.squeeze(1)
        return {"choir": udf, "contacts": contacts}
