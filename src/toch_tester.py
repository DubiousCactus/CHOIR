#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2024 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
TOCH tester.
"""

from typing import Dict, List, Optional, Tuple, Union

import torch

from model.affine_mano import AffineMANO
from src.multiview_tester import MultiViewTester
from utils import to_cuda, to_cuda_


class TOCHTester(MultiViewTester):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._data_loader.dataset.set_observations_number(1)
        self.affine_mano: AffineMANO = to_cuda_(AffineMANO(for_contactpose=True))  # type: ignore
        self._is_toch = True

    def _inference(
        self,
        samples,
        labels,
        max_observations: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        features = samples["toch_features"]
        # TODO: Assert that what I pass to the model isn't batched! Because of the structure of the code I took from the TOCH repo, I cna't use batches... We'll have to deal with this.
        verts, joints, anchors = self._model(*[f.squeeze(0) for f in features])
        return {"verts": verts, "joints": joints, "anchors": anchors}

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
        raise NotImplementedError()
