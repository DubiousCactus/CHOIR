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
from hydra_zen import (
    MISSING,
    ZenStore,
    builds,
    make_config,
    make_custom_builds_fn,
    store,
)
from metabatch import TaskLoader
from unique_names_generator import get_random_name
from unique_names_generator.data import ADJECTIVES, NAMES

from dataset.contactpose import ContactPoseDataset
from model.aggregate_cpvae import Aggregate_CPVAE
from model.baseline import BaselineModel
from src.base_trainer import BaseTrainer
from src.losses.hoi import CHOIRLoss
from src.multiview_trainer import MultiViewTrainer
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
    validation_objects: int = 3
    perturbation_level: int = 0
    noisy_samples_per_grasp: int = 50
    max_views_per_grasp: int = 5
    right_hand_only: bool = True
    center_on_object_com: bool = True
    bps_dim: int = 1024
    obj_ptcld_size: int = 3000
    debug: bool = False
    rescale: str = "pair"  # pair, fixed, none
    remap_bps_distances: bool = False
    exponential_map_w: float = 5.0


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
    num_workers: int = 2
    pin_memory: bool = True


" ================== Model ================== "
# Pre-set the group for store's model entries
model_store = store(group="model")

# Not that encoder_input_dim depend on dataset.img_dim, so we need to use a partial to set them in
# the launch_experiment function.


@dataclass
class BaselineModelConf:
    bps_dim: int
    encoder_layer_dims: Tuple[int] = (1024, 512, 256, 128)
    decoder_layer_dims: Tuple[int] = (128, 256, 512)
    latent_dim: int = 32
    predict_anchor_orientation: bool = MISSING
    predict_mano: bool = MISSING
    share_decoder_for_all_tasks: bool = True


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
        Aggregate_CPVAE,
        builds_bases=(BaselineModelConf,),
        bps_dim=MISSING,
    ),
    name="aggregate_cpvae",
)

" ================== Losses ================== "


@dataclass
class CHOIRLossConf:
    bps = MISSING
    predict_anchor_orientation: bool = False
    predict_anchor_position: bool = False
    predict_mano: bool = False
    orientation_w: float = 1.0
    distance_w: float = 1000.0
    assignment_w: float = 1.0
    mano_pose_w: float = 1.0
    mano_global_pose_w: float = 1.0
    mano_shape_w: float = 1.0
    mano_agreement_w: float = 1.0
    mano_anchors_w: float = 1.0
    kl_w: float = 1e-1
    multi_view: bool = False


training_loss_store = store(group="training_loss")
training_loss_store(
    pbuilds(
        CHOIRLoss,
        builds_bases=(CHOIRLossConf,),
    ),
    name="choir",
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
    epochs: int = 500
    seed: int = 42
    val_every: int = 1
    viz_every: int = 0
    viz_train_every: int = 0
    viz_num_samples: int = 5
    load_from_path: Optional[str] = None
    load_from_run: Optional[str] = None


training_store = store(group="training")
training_store(TrainingConfig, name="default")

trainer_store = store(group="trainer")
trainer_store(pbuilds(BaseTrainer, populate_full_signature=True), name="base")
trainer_store(pbuilds(MultiViewTrainer, populate_full_signature=True), name="multiview")


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
        {"training_loss": "choir"},
    ],
    trainer=MISSING,
    dataset=MISSING,
    model=MISSING,
    optimizer=MISSING,
    scheduler=MISSING,
    training=MISSING,
    training_loss=MISSING,
    data_loader=pbuilds(
        TaskLoader, builds_bases=(DataloaderConf,)
    ),  # Needs a partial because we need to set the dataset
)
store(Experiment, name="base_experiment")

# the experiment configs:
# - must be stored under the _global_ package
# - must inherit from `Experiment`
experiment_store = store(group="experiment", package="_global_")

experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model": "aggregate_cpvae"},
            {"override /trainer": "multiview"},
        ],
        dataset=dict(perturbation_level=2),
        training_loss=dict(multi_view=True),
        data_loader=dict(batch_size=32),
        # model=dict(encoder_layer_dims=(1024, 512, 256), decoder_layer_dims=(256, 512),
        # latent_dim=128),
        bases=(Experiment,),
    ),
    name="multiview",
)

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
        TaskLoader, builds_bases=(DataloaderConf,), shuffle=False, drop_last=False
    ),  # Needs a partial because we need to set the dataset
)
store(ExperimentEvaluation, name="base_experiment_evaluation")

# the experiment configs:
# - must be stored under the _global_ package
# - must inherit from `Experiment`
experiment_store = store(group="experiment_evaluation", package="_global_")
