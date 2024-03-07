#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

from typing import Tuple

import torch


class DDPMLoss(torch.nn.Module):
    def __init__(self, contacts_weight: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
        self.contacts_weight = contacts_weight

    def forward(
        self,
        samples: torch.Tensor,
        labels: torch.Tensor,
        model_output: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        if type(model_output) is dict:
            return {
                "udf_mse": torch.nn.functional.mse_loss(
                    model_output["udf"][0],
                    model_output["udf"][1],
                    reduction=self.reduction,
                ),
                "contacts_mse": self.contacts_weight
                * torch.nn.functional.mse_loss(
                    model_output["contacts"][0],
                    model_output["contacts"][1],
                    reduction=self.reduction,
                ),
                "ancho_obj_udf_mse": torch.nn.functional.mse_loss(
                    model_output["ancho_obj_udf"][0],
                    model_output["ancho_obj_udf"][1],
                    reduction=self.reduction,
                ),
            }
        else:
            return {
                "udf_mse": torch.nn.functional.mse_loss(
                    model_output[0], model_output[1], reduction=self.reduction
                )
            }
