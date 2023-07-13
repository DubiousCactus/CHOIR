#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Configurations for the experiments and config groups, using hydra-zen.
"""

from dataclasses import dataclass
from test import launch_test
from typing import Optional, Tuple

import torch
from hydra.conf import HydraConf, JobConf, RunDir
from hydra_zen import MISSING, ZenStore, builds, make_custom_builds_fn, store
from torch.utils.data import DataLoader
from unique_names_generator import get_random_name
from unique_names_generator.data import ADJECTIVES, NAMES

from dataset.contactpose import ContactPoseDataset
from model.baseline import BaselineModel, BaselineUNetModel
from src.base_trainer import BaseTrainer
from train import launch_experiment

# Set hydra.job.chdir=True using store():
hydra_store = ZenStore(overwrite_ok=True)
hydra_store(HydraConf(job=JobConf(chdir=True)), name="config", group="hydra")
# We'll generate a unique name for the experiment and use it as the run name
hydra_store(
    HydraConf(
        run=RunDir(
            f"runs/{get_random_name(combo=[ADJECTIVES, NAMES], separator='-', style='lowercase')}"
        )
    ),
    name="config",
    group="hydra",
)
hydra_store.add_to_hydra_store()
pbuilds = make_custom_builds_fn(zen_partial=True, populate_full_signature=False)

" ================== Dataset ================== "


# Dataclasses are a great and simple way to define a base config group with default values.
@dataclass
class GraspingDatasetConf:
    split: str = "train"
    tiny: bool = False
    augment: bool = False
    validation_objects: int = 5
    n_random_choir_per_sample: int = 100
    perturbation_level: float = 0.0
    anchor_assignment: str = "random"
    scaling: str = "none"
    unit_cube: bool = False
    positive_unit_cube: bool = False
    right_hand_only: bool = True
    bps_dim: int = 1024
    obj_ptcld_size: int = 3000
    debug: bool = False


# Pre-set the group for store's dataset entries
dataset_store = store(group="dataset")
dataset_store(
    pbuilds(ContactPoseDataset, builds_bases=(GraspingDatasetConf,)), name="contactpose"
)

" ================== Dataloader & sampler ================== "


@dataclass
class SamplerConf:
    batch_size: int = 32
    drop_last: bool = True
    shuffle: bool = True


@dataclass
class DataloaderConf:
    batch_size: int = 32
    drop_last: bool = True
    shuffle: bool = True
    num_workers: int = 8
    pin_memory: bool = True


" ================== Model ================== "
# Pre-set the group for store's model entries
model_store = store(group="model")

# Not that encoder_input_dim depend on dataset.img_dim, so we need to use a partial to set them in
# the launch_experiment function.


@dataclass
class BaselineModelConf:
    bps_dim: int
    encoder_layer_dims: Tuple[int] = (2048, 1024, 512, 256)
    decoder_layer_dims: Tuple[int] = (256, 512, 1024)
    latent_dim: int = 128
    predict_anchor_orientation: bool = True
    predict_mano: bool = True


model_store(
    pbuilds(
        BaselineModel,
        builds_bases=(BaselineModelConf,),
        bps_dim=MISSING,
    ),
    name="baseline",
)
model_store(
    pbuilds(
        BaselineUNetModel,
        builds_bases=(BaselineModelConf,),
        bps_dim=MISSING,
    ),
    name="baseline_unet",
)

" ================== Optimizer ================== "


@dataclass
class Optimizer:
    lr: float = 1e-3
    weight_decay: float = 0.0


opt_store = store(group="optimizer")
opt_store(
    pbuilds(
        torch.optim.Adam,
        builds_bases=(Optimizer,),
    ),
    name="adam",
)
opt_store(
    pbuilds(
        torch.optim.SGD,
        builds_bases=(Optimizer,),
    ),
    name="sgd",
)


" ================== Scheduler ================== "
sched_store = store(group="scheduler")
sched_store(
    pbuilds(
        torch.optim.lr_scheduler.StepLR,
        step_size=100,
        gamma=0.5,
    ),
    name="step",
)
sched_store(
    pbuilds(
        torch.optim.lr_scheduler.ReduceLROnPlateau,
        mode="min",
        factor=0.5,
        patience=10,
    ),
    name="plateau",
)
sched_store(
    pbuilds(
        torch.optim.lr_scheduler.CosineAnnealingLR,
    ),
    name="cosine",
)

" ================== Experiment ================== "


@dataclass
class TrainingConfig:
    epochs: int = 200
    seed: int = 42
    val_every: int = 1
    viz_every: int = 10
    viz_train_every: int = 0
    load_from_path: Optional[str] = None
    load_from_run: Optional[str] = None


training_store = store(group="training")
training_store(TrainingConfig, name="default")

trainer_store = store(group="trainer")
trainer_store(pbuilds(BaseTrainer, populate_full_signature=True), name="base")


Experiment = builds(
    launch_experiment,
    populate_full_signature=True,
    hydra_defaults=[
        "_self_",
        {"trainer": "base"},
        {"dataset": "contactpose"},
        {"model": "baseline"},
        {"optimizer": "adam"},
        {"scheduler": "step"},
        {"training": "default"},
    ],
    trainer=MISSING,
    dataset=MISSING,
    model=MISSING,
    optimizer=MISSING,
    scheduler=MISSING,
    training=MISSING,
    data_loader=pbuilds(
        DataLoader, builds_bases=(DataloaderConf,)
    ),  # Needs a partial because we need to set the dataset
)
store(Experiment, name="base_experiment")

# the experiment configs:
# - must be stored under the _global_ package
# - must inherit from `Experiment`
experiment_store = store(group="experiment", package="_global_")

" ================== Model testing ================== "


@dataclass
class TestingConfig:
    seed: int = 42
    viz_every: int = 10
    load_from_path: Optional[str] = None
    load_from_run: Optional[str] = None


training_store = store(group="testing")
training_store(TestingConfig, name="default")


ExperimentEvaluation = builds(
    launch_test,
    populate_full_signature=True,
    hydra_defaults=[
        "_self_",
        {"dataset": "contactpose"},
        {"model": "baseline"},
        {"testing": "default"},
    ],
    dataset=MISSING,
    model=MISSING,
    testing=MISSING,
    data_loader=pbuilds(
        DataLoader, builds_bases=(DataloaderConf,), shuffle=False, drop_last=False
    ),  # Needs a partial because we need to set the dataset
)
store(ExperimentEvaluation, name="base_experiment_evaluation")

# the experiment configs:
# - must be stored under the _global_ package
# - must inherit from `Experiment`
experiment_store = store(group="experiment_evaluation", package="_global_")
