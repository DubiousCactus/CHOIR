#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Project-level constants: you would typically configure these once for your project. Avoid putting
experiment-specific constants here.
"""

from enum import Enum
from os import environ as env

from utils import str_to_bool


class TerminationBehavior(Enum):
    WAIT_FOR_EPOCH_END = 0
    ABORT_EPOCH = 1


""" Project-level constants
- *str_to_bool* accepts "1", "true", "yes", "t" as True, and "0", "false", "no", "f" as False.
- *env.get("VAR_NAME", "default_value")* returns the value of the environment variable VAR_NAME if
                                          it exists, or "default_value" otherwise.
"""
REPRODUCIBLE = str_to_bool(env.get("REPRODUCIBLE", "True"))
CKPT_PATH = "ckpt"
USE_CUDA_IF_AVAILABLE = str_to_bool(env.get("USE_CUDA_IF_AVAILABLE", "True"))
SIGINT_BEHAVIOR = TerminationBehavior.WAIT_FOR_EPOCH_END
PARTIALLY_LOAD_MODEL_IF_NO_FULL_MATCH = str_to_bool(
    env.get("PARTIALLY_LOAD_MODEL_IF_NO_FULL_MATCH", "t")
)
BEST_N_MODELS_TO_KEEP = int(
    env.get("BEST_N_MODELS_TO_KEEP", 5)
)  # 0 means keep all models
USE_WANDB = str_to_bool(env.get("USE_WANDB", "true"))
PROJECT_NAME = "my-python-project"
PLOT_ENABLED = str_to_bool(env.get("PLOT_ENABLED", "1"))
HEADLESS = str_to_bool(env.get("HEADLESS", "0"))

# Theming
class Theme(Enum):
    TRAINING = "green"
    VALIDATION = "blue"
    TESTING = "cyan"

    def __str__(self):
        return f"{self.value}"

    def __repr__(self):
        return f"{self.value}"


ANSI_COLORS = {
    "black": 30,
    "red": 31,
    "green": 32,
    "yellow": 33,
    "blue": 34,
    "magenta": 35,
    "cyan": 36,
    "white": 37,
}
