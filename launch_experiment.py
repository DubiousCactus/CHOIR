#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Training code.
"""

import os

import hydra_zen
import torch
import yaml
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from hydra_zen import just
from hydra_zen.typing import Partial

import conf.experiment  # Must import the config to add all components to the store!
import wandb
from conf import project as project_conf
from src.base_trainer import BaseTrainer
from utils import colorize, to_cuda_


def launch_experiment(
    run,
    data_loader: Partial[torch.utils.data.DataLoader],
    optimizer: Partial[torch.optim.Optimizer],
    scheduler: Partial[torch.optim.lr_scheduler._LRScheduler],
    trainer: Partial[BaseTrainer],
    tester: Partial[BaseTrainer],
    dataset: torch.utils.data.Dataset,
    model: Partial[torch.nn.Module],
    training_loss: Partial[torch.nn.Module],
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
    exp_conf = hydra_zen.to_yaml(
        dict(
            run_name=run_name,
            run_conf=run,
            dataset=dataset,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            training_loss=training_loss,
        )
    )
    print(
        colorize(
            "Experiment config:\n" + "_" * 18 + "\n" + exp_conf + "_" * 18, color_code
        )
    )

    "============ Partials instantiation ============"
    model_inst = model(
        bps_dim=just(dataset).bps_dim,
        remapped_bps_distances=just(
            dataset
        ).remap_bps_distances,  # Whether to use sigmoid in last layer
        predict_anchor_orientation=just(training_loss).predict_anchor_orientation
        or just(training_loss).predict_anchor_position,
        predict_mano=just(training_loss).predict_mano,
    )  # Use just() to get the config out of the Zen-Partial
    print(model_inst)
    print(f"Number of parameters: {sum(p.numel() for p in model_inst.parameters())}")
    print(
        f"Number of trainable parameters: {sum(p.numel() for p in model_inst.parameters() if p.requires_grad)}"
    )
    train_dataset, val_dataset, test_dataset = (
        dataset(split="train"),
        dataset(split="val"),
        dataset(split="test"),
    )
    opt_inst = optimizer(model_inst.parameters())
    scheduler_inst = scheduler(
        opt_inst
    )  # TODO: less hacky way to set T_max for CosineAnnealingLR?
    if isinstance(scheduler_inst, torch.optim.lr_scheduler.CosineAnnealingLR):
        scheduler_inst.T_max = run.epochs

    training_loss_inst = training_loss(
        bps=train_dataset.bps,
        remap_bps_distances=train_dataset.remap_bps_distances,
        exponential_map_w=train_dataset.exponential_map_w,
    )

    "============ CUDA ============"
    model_inst: torch.nn.Module = to_cuda_(model_inst)  # type: ignore
    training_loss_inst: torch.nn.Module = to_cuda_(training_loss_inst)  # type: ignore
    # model_inst = torch.compile(model_inst)

    "============ Weights & Biases ============"
    if project_conf.USE_WANDB:
        # exp_conf is a string, so we need to load it back to a dict:
        exp_conf = yaml.safe_load(exp_conf)
        wandb.init(
            project=project_conf.PROJECT_NAME,
            name=run_name,
            config=exp_conf,
        )
        wandb.watch(model_inst, log="all", log_graph=True)
    " ============ Reproducibility of data loaders ============ "
    g = None
    if project_conf.REPRODUCIBLE:
        g = torch.Generator()
        g.manual_seed(run.seed)

    train_loader_inst = data_loader(train_dataset, generator=g)
    val_loader_inst = data_loader(
        val_dataset, generator=g, shuffle=False, drop_last=False
    )
    test_loader_inst = data_loader(
        test_dataset, generator=g, shuffle=False, drop_last=False
    )

    " ============ Training ============ "
    model_ckpt_path = None

    if run.load_from_run is not None and run.load_from_path is not None:
        raise ValueError(
            "Both training.load_from_path and training.load_from_run are set. Please choose only one."
        )
    elif run.load_from_run is not None:
        run_models = sorted(
            [
                f
                for f in os.listdir(to_absolute_path(f"runs/{run.load_from_run}/"))
                if f.endswith(".ckpt")
                and (not f.startswith("last") if not run.training_mode else True)
            ]
        )
        if len(run_models) < 1:
            raise ValueError(f"No model found in runs/{run.load_from_run}/")
        model_ckpt_path = to_absolute_path(
            os.path.join(
                "runs",
                run.load_from_run,
                run_models[-1],
            )
        )
    elif run.load_from_path is not None:
        model_ckpt_path = to_absolute_path(run.load_from_path)

    if run.training_mode:
        trainer(
            run_name=run_name,
            model=model_inst,
            opt=opt_inst,
            scheduler=scheduler_inst,
            train_loader=train_loader_inst,
            val_loader=val_loader_inst,
            training_loss=training_loss_inst,
        ).train(
            epochs=run.epochs,
            val_every=run.val_every,
            visualize_every=run.viz_every,
            visualize_train_every=run.viz_train_every,
            visualize_n_samples=run.viz_num_samples,
            model_ckpt_path=model_ckpt_path,
        )
    else:
        tester(
            run_name=run_name,
            model=model_inst,
            data_loader=test_loader_inst,
            model_ckpt_path=model_ckpt_path,
            training_loss=training_loss_inst,
        ).test(
            visualize_every=run.viz_every,
        )
