#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Training utilities. This is a good place for your code that is used in training (i.e. custom loss
function, visualization code, etc.)
"""

from typing import List, Tuple, Union

import torch

import conf.project as project_conf
from utils import to_cuda_


def visualize_model_predictions(
    model: torch.nn.Module, batch: Union[Tuple, List, torch.Tensor], step: int
) -> None:
    x, y = to_cuda_(batch)  # type: ignore
    if not project_conf.HEADLESS:
        raise NotImplementedError(
            "Visualization is not implemented for the headless mode."
        )
    if project_conf.USE_WANDB:
        # TODO: Log a few predictions and the ground truth to wandb.
        # wandb.log({"pointcloud": wandb.Object3D(ptcld)}, step=step)
        raise NotImplementedError("Visualization is not implemented for wandb.")
