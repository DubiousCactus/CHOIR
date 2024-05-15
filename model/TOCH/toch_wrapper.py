#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2024 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
TOCH wrapper around the model but w/ TTO as well, only for evaluation.
"""


import torch

import conf.project as project_conf
from model.affine_mano import AffineMANO
from utils import colorize


class TOCHInference(torch.nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.single_modality = "noisy_pair"
        print(
            colorize(
                "Creating TOCH inference wrapper for ContactPose only!",
                project_conf.ANSI_COLORS["red"],
            )
        )
        self.affine_mano = AffineMANO(for_contactpose=True)

    def forward(self):
        # What to return (bare minimum):
        # { "verts": , "joints": , "anchors": }
        # I think the anchors can be ignored.
        pass
