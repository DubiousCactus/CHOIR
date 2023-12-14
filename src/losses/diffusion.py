#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

from typing import Tuple

import torch


class DDPMLoss(torch.nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        samples: torch.Tensor,
        labels: torch.Tensor,
        model_output: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        return {
            "mse": torch.nn.functional.mse_loss(
                model_output[0], model_output[1], reduction=self.reduction
            )
        }
