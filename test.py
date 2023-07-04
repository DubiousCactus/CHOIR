#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Testing code.
"""

import os

import hydra_zen
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from hydra_zen import just, store, zen
from hydra_zen.typing import Partial

import conf.experiment  # Must import the config to add all components to the store!
from conf import project as project_conf
from src.base_tester import BaseTester
from utils import colorize, seed_everything, to_cuda_


def launch_test(
    testing,
    dataset: torch.utils.data.Dataset,
    data_loader: Partial[torch.utils.data.DataLoader],
    model: Partial[torch.nn.Module],
):
    run_name = os.path.basename(HydraConfig.get().runtime.output_dir)
    # Generate a random ANSI code:
    color_code = f"38;5;{hash(run_name) % 255}"
    print(
        colorize(
            f"========================= Running {run_name} =========================",
            color_code,
        )
    )

    "============ Partials instantiation ============"
    model_inst = model(
        encoder_input_dim=just(dataset).img_dim ** 2
    )  # Use just() to get the config out of the Zen-Partial
    test_dataset = dataset(split="test")

    "============ CUDA ============"
    model_inst: torch.nn.Module = to_cuda_(model_inst)  # type: ignore

    " ============ Reproducibility of data loaders ============ "
    g = None
    if project_conf.REPRODUCIBLE:
        g = torch.Generator()
        g.manual_seed(testing.seed)

    test_loader_inst = data_loader(test_dataset, generator=g)

    " ============ Testing ============ "
    model_ckpt_path = None

    if testing.load_from_run is not None and testing.load_from_path is not None:
        raise ValueError(
            "Both testing.load_from_path and testing.load_from_run are set. Please choose only one."
        )
    elif testing.load_from_run is not None:
        run_models = sorted(
            [
                f
                for f in os.listdir(to_absolute_path(f"runs/{testing.load_from_run}/"))
                if f.endswith(".ckpt")
            ]
        )
        if len(run_models) < 1:
            raise ValueError(f"No model found in runs/{testing.load_from_run}/")
        model_ckpt_path = to_absolute_path(
            os.path.join(
                "runs",
                testing.load_from_run,
                run_models[-1],
            )
        )
    elif testing.load_from_path is not None:
        model_ckpt_path = to_absolute_path(testing.load_from_path)
    else:
        raise ValueError(
            "No model to load. Please set either testing.load_from_path or testing.load_from_run."
        )

    BaseTester(
        run_name=run_name,
        model=model_inst,
        data_loader=test_loader_inst,
        model_ckpt_path=model_ckpt_path,
    ).test(
        visualize_every=testing.viz_every,
    )


if __name__ == "__main__":
    "============ Hydra-Zen ============"
    store.add_to_hydra_store(
        overwrite_ok=True
    )  # Overwrite Hydra's default config to update it
    zen(
        launch_test,
        pre_call=hydra_zen.zen(
            lambda testing: seed_everything(
                testing.seed
            )  # testing is the config of the testing group, part of the base config
            if project_conf.REPRODUCIBLE
            else lambda: None
        ),
    ).hydra_main(
        config_name="base_experiment_evaluation",
        version_base="1.3",  # Hydra base version
    )
