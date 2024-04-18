#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2024 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Motion Sequence DDPM Tester.
"""


from typing import Dict, Optional

import torch

from src.multiview_tester import MultiViewTester


class SeqDDPMTester(MultiViewTester):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conditional = kwargs.get("conditional", False)
        self._data_loader.dataset.set_observations_number(1)
        self._full_choir = kwargs.get("full_choir", False)
        self._use_deltas = self._data_loader.dataset.use_deltas
        self._model_contacts = kwargs.get("model_contacts", False)
        self._enable_contacts_tto = kwargs.get("enable_contacts_tto", False)
        self._use_ema = kwargs.get("use_ema", False)
        self._n_augmentations = 10
        if self._model_contacts:
            self._model.backbone.set_anchor_indices(
                self._data_loader.dataset.anchor_indices
            )
            self._model.set_dataset_stats(self._data_loader.dataset)
            if self._ema is not None:
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
            if self._ema is not None:
                self._ema.ema_model = torch.compile(self._ema.ema_model)

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
            modality = "noisy_pair" if self._inference_mode == "denoising" else "object"
        print(
            f"y shape: {samples['choir'].shape}, max_observations: {max_observations}"
        )
        y = (
            samples["choir"][:, :1] if self.conditional else None
        )  # We're using only one frame
        print(f"[*] Using modality: {modality}")
        if modality == "object":
            y = y[..., 0].unsqueeze(-1)
        elif modality == "noisy_pair":
            # Already comes in noisy_pair modality
            pass
        udf, contacts = model.generate(
            1,
            y=y,
            y_modality=modality,
        )
        rotations = None
        # Only use 1 sample for now. TODO: use more samples and average?
        # No, I will do what GraspTTA is doing: apply N random rotations to the object and draw one
        # sample. It's the fairest way to compare since GraspTTA can't get any grasp variations
        # from just sampling z (unlike ours hehehe).
        # TODO: Apply augmentations to the test dataset (can't rotate CHOIRs easily).
        udf, contacts = udf.squeeze(1), contacts.squeeze(1)
        return {"choir": udf, "contacts": contacts, "rotations": rotations}
