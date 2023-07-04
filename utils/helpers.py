#! /usr/bin/env python
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Some utility classes and functions for your convenience.
"""

import os
import os.path as osp
from collections.abc import Callable
from typing import Any

from hydra.core.hydra_config import HydraConfig


class BestNModelSaver:
    def __init__(self, n: int, save_callback: Callable) -> None:
        self._n = n
        self._best_n_models = {}
        self._save_callback = save_callback
        self._min_val_loss = float("inf")
        self._min_val_loss_epoch: int = 0

    def __call__(self, epoch: int, val_loss: float) -> Any:
        if val_loss < self._min_val_loss:
            ckpt_path = osp.join(
                HydraConfig.get().runtime.output_dir,
                f"epoch_{epoch:03d}_val-loss_{val_loss:06f}.ckpt",
            )
            self._save_if_best_model(
                val_loss, ckpt_path
            ) if self._n > 0 else self._save_callback(val_loss, ckpt_path)
            self._min_val_loss = val_loss
            self._min_val_loss_epoch = epoch

    def _save_if_best_model(self, val_loss: float, ckpt_path: str):
        if len(self._best_n_models) < self._n or val_loss < max(
            self._best_n_models.keys()
        ):
            if len(self._best_n_models) == self._n:
                worst_of_best = max(self._best_n_models.keys())
                del_fname = self._best_n_models[worst_of_best]
                os.remove(del_fname)
                del self._best_n_models[worst_of_best]
            self._best_n_models[val_loss] = ckpt_path
            self._save_callback(val_loss, ckpt_path)

    @property
    def min_val_loss(self):
        return self._min_val_loss

    @property
    def min_val_loss_epoch(self):
        return self._min_val_loss_epoch
