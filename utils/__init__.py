#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.


import random
from typing import Any, List, Tuple, Union

import numpy as np
import torch
import tqdm

from conf import project as project_conf


def str_to_bool(s: str) -> bool:
    assert s.lower() in (
        "yes",
        "true",
        "t",
        "1",
        "no",
        "false",
        "f",
        "0",
    ), f"Invalid boolean value: {s}"
    return s.lower() in ("yes", "true", "t", "1")


def seed_everything(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def to_cuda_(x: Any) -> Union[Tuple, List, torch.Tensor, torch.nn.Module]:
    if project_conf.USE_CUDA_IF_AVAILABLE and torch.cuda.is_available():
        if isinstance(x, torch.Tensor) or isinstance(x, torch.nn.Module):
            x = x.cuda().float()
        elif isinstance(x, tuple):
            x = tuple(to_cuda_(t) for t in x)
        elif isinstance(x, list):
            x = [to_cuda_(t) for t in x]
        elif isinstance(x, dict):
            x = {key: to_cuda_(value) for key, value in x.items()}
    return x


def to_cuda(func):
    """Decorator to move function arguments to cuda if available and if they are
    torch tensors, torch modules or tuples/lists of."""

    def wrapper(*args, **kwargs):
        args = to_cuda_(args)
        for key, value in kwargs.items():
            kwargs[key] = to_cuda_(value)
        return func(*args, **kwargs)

    return wrapper


def colorize(string: str, ansii_code: Union[int, str]) -> str:
    return f"\033[{ansii_code}m{string}\033[0m"


def blink_pbar(i: int, pbar: tqdm.tqdm, n: int) -> None:
    """Blink the progress bar every n iterations.
    Args:
        i (int): current iteration
        pbar (tqdm.tqdm): progress bar
        n (int): blink every n iterations
    """
    if i % n == 0:
        pbar.colour = (
            project_conf.Theme.TRAINING.value
            if pbar.colour == project_conf.Theme.VALIDATION.value
            else project_conf.Theme.VALIDATION.value
        )


def update_pbar_str(pbar: tqdm.tqdm, string: str, color_code: int) -> None:
    """Update the progress bar string.
    Args:
        pbar (tqdm.tqdm): progress bar
        string (str): string to update the progress bar with
        color_code (int): color code for the string
    """
    pbar.set_description_str(colorize(string, color_code))
