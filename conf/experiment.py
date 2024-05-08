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

import conf.project as project_conf
from dataset.contactpose import ContactPoseDataset
from dataset.grab import GRABDataset
from dataset.oakink import OakInkDataset
from launch_experiment import launch_experiment
from model.aggregate_ved import Aggregate_VED
from model.baseline import BaselineModel
from model.diffusion_model import (
    BPSDiffusionModel,
    ContactsBPSDiffusionModel,
    KPContactsDiffusionModel,
    KPDiffusionModel,
)
from model.graspTTA.affordanceNet_obman_mano_vertex import affordanceNet
from model.graspTTA.ContactNet import pointnet_reg
from model.graspTTA.grasptta_wrapper import GraspTTA
from src.base_tester import BaseTester
from src.base_trainer import BaseTrainer
from src.ddpm_tester import DDPMTester
from src.ddpm_trainer import DDPMTrainer
from src.grasptta_contactnet_trainer import ContactNetTrainer
from src.grasptta_graspcvae_tester import GraspCVAETester
from src.grasptta_graspcvae_trainer import GraspCVAETrainer
from src.grasptta_tester import GraspTTATester
from src.losses.diffusion import DDPMLoss
from src.losses.graspTTA import GraspCVAELoss
from src.losses.hoi import CHOIRLoss
from src.multiview_ddpm_baseline_tester import MultiViewDDPMBaselineTester
from src.multiview_ddpm_baseline_trainer import MultiViewDDPMBaselineTrainer
from src.multiview_ddpm_tester import MultiViewDDPMTester
from src.multiview_ddpm_trainer import MultiViewDDPMTrainer
from src.multiview_tester import MultiViewTester
from src.multiview_trainer import MultiViewTrainer
from src.seq_ddpm_tester import SeqDDPMTester

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
    n_augs: int = 10
    validation_objects: int = 3
    test_objects: int = 2
    perturbation_level: int = 0
    min_views_per_grasp: int = 1
    max_views_per_grasp: int = 5
    right_hand_only: bool = True
    center_on_object_com: bool = True
    bps_dim: int = 1024
    obj_ptcld_size: int = 10000
    debug: bool = False
    rescale: str = "none"  # pair, fixed, none
    remap_bps_distances: bool = True
    exponential_map_w: float = 5.0
    random_anchor_assignment: bool = True
    use_deltas: bool = False
    use_bps_grid: bool = False


# Pre-set the group for store's dataset entries
dataset_store = store(group="dataset")
dataset_store(
    pbuilds(
        ContactPoseDataset,
        builds_bases=(GraspingDatasetConf,),
        noisy_samples_per_grasp=16,
        use_contactopt_splits=False,
        use_improved_contactopt_splits=False,
        eval_observations_plateau=False,
        eval_anchor_assignment=False,
        load_pointclouds=False,
        model_contacts=False,
    ),
    name="contactpose",
)
dataset_store(
    pbuilds(
        OakInkDataset,
        builds_bases=(GraspingDatasetConf,),
        dataset_root="/home/moralest/ssd_storage/OakInkData",
        model_contacts=True,
        noisy_samples_per_grasp=5,
    ),
    name="oakink",
)
dataset_store(
    pbuilds(
        GRABDataset,
        builds_bases=(GraspingDatasetConf,),
        root_path="/Users/cactus/Code/GRAB",
        smplx_path=project_conf.SMPLX_MODEL_PATH,
        use_affine_mano=False,
        use_official_splits=True,
        perturbation_level=1,
        static_grasps_only=False,
    ),
    name="grab",
)

" ================== Dataloader & sampler ================== "


@dataclass
class SamplerConf:
    batch_size: int = 64
    drop_last: bool = True
    shuffle: bool = True


@dataclass
class DataloaderConf:
    batch_size: int = 64
    drop_last: bool = True
    shuffle: bool = True
    num_workers: int = 4
    pin_memory: bool = False
    n_batches: Optional[int] = None
    prefetch_factor: Optional[int] = None
    persistent_workers: bool = True


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
        Aggregate_VED,
        builds_bases=(BaselineModelConf,),
        bps_dim=MISSING,
        remapped_bps_distances=MISSING,  # Sigmoid if so
        batch_norm=True,
        decoder_use_obj=False,
        skip_connections=True,
        residual_connections=True,
        encoder_dropout=False,
        decoder_dropout=False,
        predict_deltas=False,
        frame_to_predict=MISSING,
        encoder_layer_dims=(512, 512, 512),
        decoder_layer_dims=(1024, 1024, 1024),
        choir_encoder_dims=(2048, 2048, 2048, 2048),
        choir_embedding_dim=256,
        aggregator="mean",
        agg_heads=8,
        agg_kq_dim=1024,
    ),
    name="agg_ved",
)


model_store(
    pbuilds(
        BPSDiffusionModel,
        backbone="mlp_resnet",
        time_steps=1000,
        beta_1=1e-4,
        beta_T=0.02,
        bps_dim=MISSING,
        choir_dim=MISSING,
        rescale_input=MISSING,
        temporal_dim=256,
        y_embed_dim=256,
        context_channels=MISSING,
        embed_full_choir=False,
        use_backbone_self_attn=False,
        use_encoder_self_attn=False,
    ),
    name="bps_ddpm",
)

model_store(
    pbuilds(
        ContactsBPSDiffusionModel,
        backbone="3d_unet",
        time_steps=1000,
        beta_1=1e-4,
        beta_T=0.02,
        bps_dim=MISSING,
        temporal_dim=256,
        y_embed_dim=256,
        context_channels=MISSING,
        use_backbone_self_attn=False,
        use_encoder_self_attn=False,
        object_in_encoder=False,
        contacts_hidden_dim=2048,
        contacts_skip_connections=False,
        input_normalization="standard",
        conv_transpose=True,
        single_modality=None,
    ),
    name="contact_bps_ddpm",
)


model_store(
    pbuilds(
        KPDiffusionModel,
        backbone="mlp_resnet",
        time_steps=1000,
        beta_1=1e-4,
        beta_T=0.02,
        n_hand_keypoints=21 + 32,  # 21 MANO joints + 32 contact anchors
        n_obj_keypoints=MISSING,
        rescale_input=MISSING,
        temporal_dim=256,
        y_embed_dim=256,
        y_input_keypoints=MISSING,
        object_in_encoder=True,
        skip_connections=False,
    ),
    name="kp_ddpm",
)

model_store(
    pbuilds(
        KPContactsDiffusionModel,
        backbone="mlp_resnet",
        time_steps=1000,
        beta_1=1e-4,
        beta_T=0.02,
        n_hand_keypoints=21 + 32,  # 21 MANO joints + 32 contact anchors
        n_obj_keypoints=MISSING,
        temporal_dim=256,
        y_embed_dim=256,
        y_input_keypoints=MISSING,
        object_in_encoder=False,
        skip_connections=True,
        single_modality="noisy_pair",
    ),
    name="kp_coddpm",
)

model_store(
    pbuilds(
        affordanceNet,
        mano_params_dim=MISSING,
    ),
    name="grasp_tta_cvae",
)

model_store(
    pbuilds(
        pointnet_reg,
        num_class=1,
        with_rgb=False,
        n_pts=3000,
    ),
    name="grasp_tta_contactnet",
)

model_store(
    pbuilds(
        GraspTTA,
        mano_params_dim=MISSING,
        num_class=1,
        with_rgb=False,
        n_pts=3000,
        tto_steps=300,
        graspCVAE_model_pth=None,
        contactnet_model_pth=None,
    ),
    name="grasp_tta_wrapper",
)

" ================== Losses ================== "


@dataclass
class CHOIRLossConf:
    bps = MISSING
    remap_bps_distances: bool = MISSING
    exponential_map_w: float = MISSING
    predict_anchor_orientation: bool = False
    predict_anchor_position: bool = False
    predict_mano: bool = False
    orientation_w: float = 1.0
    distance_w: float = 1.0
    assignment_w: float = 1.0
    mano_pose_w: float = 1.0
    mano_global_pose_w: float = 1.0
    mano_shape_w: float = 1.0
    mano_agreement_w: float = 1.0
    mano_anchors_w: float = 1.0
    kl_w: float = 1e-4
    multi_view: bool = False
    temporal: bool = False
    use_kl_scheduler: bool = False


training_loss_store = store(group="training_loss")
training_loss_store(
    pbuilds(
        CHOIRLoss,
        builds_bases=(CHOIRLossConf,),
    ),
    name="choir",
)

training_loss_store(
    pbuilds(
        DDPMLoss,
        contacts_weight=0.5,
        reduction="mean",
    ),
    name="diffusion",
)

training_loss_store(
    pbuilds(
        GraspCVAELoss,
        hand_weights_path="src/losses/rhand_weights.npy",
    ),
    name="grasp_tta_cvae",
)
" ================== Optimizer ================== "


@dataclass
class Optimizer:
    lr: float = 3e-4
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
        torch.optim.AdamW,
        builds_bases=(Optimizer,),
        weight_decay=1e-2,
    ),
    name="adamw",
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
        gamma=0.7,
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
sched_store(
    pbuilds(torch.optim.lr_scheduler.ExponentialLR, gamma=0.999),
    name="exp",
)

" ================== Experiment ================== "


@dataclass
class RunConfig:
    epochs: int = 1000
    seed: int = 42
    val_every: int = 1
    viz_every: int = 0
    viz_train_every: int = 0
    viz_num_samples: int = 5
    load_from: Optional[str] = None
    training_mode: bool = True
    fine_tune: bool = False
    save_predictions: bool = False
    max_observations: int = 1  # For testing
    # As ideas changed and I had to try many things quickly, I became increasingly lazy and this
    # bullshit is the result:
    conditional: bool = False
    full_choir: bool = False
    disable_grad: bool = False
    model_contacts: bool = False
    enable_contacts_tto: bool = True
    use_ema: bool = False
    compute_iv: bool = True
    compute_sim_displacement: bool = True
    compile_test_model: bool = False
    compute_contact_scores: bool = True
    compute_pose_error: bool = True
    debug_tto: bool = False
    dump_videos: bool = False
    inference_mode: str = "denoising"  # "denoising" or "generation"
    # RunConfig was never meant to be soiled like this :'(


run_store = store(group="run")
run_store(RunConfig, name="default")

trainer_store = store(group="trainer")
trainer_store(pbuilds(BaseTrainer, populate_full_signature=True), name="base")
trainer_store(pbuilds(MultiViewTrainer, populate_full_signature=True), name="multiview")
trainer_store(pbuilds(DDPMTrainer, populate_full_signature=True), name="ddpm")
trainer_store(
    pbuilds(MultiViewDDPMTrainer, populate_full_signature=True), name="ddpm_multiview"
)
trainer_store(
    pbuilds(MultiViewDDPMBaselineTrainer, populate_full_signature=True),
    name="ddpm_baseline_multiview",
)
trainer_store(
    pbuilds(GraspCVAETrainer, populate_full_signature=True),
    name="grasp_tta_cvae",
)
trainer_store(
    pbuilds(ContactNetTrainer, populate_full_signature=True),
    name="grasp_tta_contactnet",
)

tester_store = store(group="tester")
tester_store(pbuilds(BaseTester, populate_full_signature=True), name="base")
tester_store(pbuilds(MultiViewTester, populate_full_signature=True), name="multiview")
tester_store(pbuilds(DDPMTester, populate_full_signature=True), name="ddpm")
tester_store(
    pbuilds(MultiViewDDPMTester, populate_full_signature=True), name="ddpm_multiview"
)
tester_store(
    pbuilds(MultiViewDDPMBaselineTester, populate_full_signature=True),
    name="ddpm_baseline_multiview",
)
tester_store(
    pbuilds(GraspTTATester, populate_full_signature=True),
    name="grasp_tta",
)
tester_store(
    pbuilds(GraspCVAETester, populate_full_signature=True),
    name="grasp_tta_cvae",
)
tester_store(
    pbuilds(SeqDDPMTester, populate_full_signature=True),
    name="seq_ddpm",
)

Experiment = builds(
    launch_experiment,
    populate_full_signature=True,
    hydra_defaults=[
        "_self_",
        {"trainer": "base"},
        {"tester": "base"},
        {"dataset": "contactpose"},
        {"model": "baseline"},
        {"optimizer": "adam"},
        {"scheduler": "step"},
        {"run": "default"},
        {"training_loss": "choir"},
    ],
    trainer=MISSING,
    tester=MISSING,
    dataset=MISSING,
    model=MISSING,
    optimizer=MISSING,
    scheduler=MISSING,
    run=MISSING,
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
            {"override /model": "bps_ddpm"},
            {"override /dataset": "contactpose"},
            {"override /trainer": "ddpm"},
            {"override /tester": "ddpm"},
            {"override /training_loss": "diffusion"},
        ],
        dataset=dict(perturbation_level=0),
        data_loader=dict(batch_size=64),
        model=dict(
            temporal_dim=256,
            y_embed_dim=None,
            choir_dim=1,
            rescale_input=True,
        ),
        bases=(Experiment,),
    ),
    name="ddpm",
)

experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model": "bps_ddpm"},
            {"override /dataset": "contactpose"},
            {"override /trainer": "ddpm"},
            {"override /tester": "ddpm"},
            {"override /training_loss": "diffusion"},
        ],
        dataset=dict(perturbation_level=0, remap_bps_distances=False, use_deltas=True),
        data_loader=dict(batch_size=64),
        model=dict(
            y_embed_dim=None,
            choir_dim=3,
            rescale_input=False,
        ),
        bases=(Experiment,),
    ),
    name="ddpm_deltas",
)

experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model": "bps_ddpm"},
            {"override /dataset": "contactpose"},
            {"override /trainer": "ddpm"},
            {"override /tester": "ddpm"},
            {"override /training_loss": "diffusion"},
        ],
        dataset=dict(
            perturbation_level=0,
            remap_bps_distances=True,
            use_deltas=False,
            use_bps_grid=True,
            bps_dim=16**3,  # 32**3 should give much better results
        ),
        data_loader=dict(batch_size=64),
        model=dict(
            y_embed_dim=None,
            choir_dim=1,
            rescale_input=True,
            backbone="3d_unet",
        ),
        bases=(Experiment,),
    ),
    name="ddpm_3d",
)

experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model": "bps_ddpm"},
            {"override /dataset": "contactpose"},
            {"override /trainer": "ddpm"},
            {"override /tester": "ddpm"},
            {"override /training_loss": "diffusion"},
        ],
        dataset=dict(
            perturbation_level=0,
            remap_bps_distances=True,
            use_deltas=False,
            use_bps_grid=True,
            bps_dim=16**3,  # 32**3 should give much better results
        ),
        data_loader=dict(batch_size=64),
        model=dict(
            y_embed_dim=256,
            choir_dim=1,
            rescale_input=True,
            backbone="3d_unet",
        ),
        run=dict(conditional=True),
        bases=(Experiment,),
    ),
    name="cddpm_3d",
)
experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model": "kp_ddpm"},
            {"override /dataset": "contactpose"},
            {"override /trainer": "ddpm_baseline_multiview"},
            {"override /tester": "ddpm_baseline_multiview"},
            {"override /training_loss": "diffusion"},
        ],
        dataset=dict(
            perturbation_level=2,
            max_views_per_grasp=1,
            use_contactopt_splits=False,
            use_improved_contactopt_splits=True,
            remap_bps_distances=True,
            use_deltas=False,
            use_bps_grid=False,  # We won't exploit it but then it's a fairer comparison
            bps_dim=1024,  # 4096 points, as used in the PointNet++ paper
            augment=True,
            n_augs=20,
        ),
        data_loader=dict(batch_size=64),
        model=dict(
            n_obj_keypoints=1024,  # 1024 points taken from the object point cloud's target points of the BPS representation
            y_input_keypoints=1024
            + 21
            + 32,  # 1024 points + 21 MANO joints + 32 contact anchors
            y_embed_dim=256,
            rescale_input=False,  # Should already be around [-1, 1]... Could be outside?
            object_in_encoder=False,
            skip_connections=False,
        ),
        run=dict(
            conditional=True, full_choir=False  # Must be equal to object_in_encoder!
        ),  # We can reuse the "full_choir" flag for "hand_object_pair"
        bases=(Experiment,),
    ),
    name="baseline_cddpm_3d_multiview_contactopt",
)
experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model": "kp_coddpm"},
            {"override /dataset": "contactpose"},
            {"override /trainer": "ddpm_baseline_multiview"},
            {"override /tester": "ddpm_baseline_multiview"},
            {"override /training_loss": "diffusion"},
        ],
        dataset=dict(
            perturbation_level=2,
            max_views_per_grasp=1,
            use_contactopt_splits=False,
            use_improved_contactopt_splits=True,
            remap_bps_distances=True,
            use_deltas=False,
            use_bps_grid=True,  # We won't exploit it but then it's a fairer comparison
            bps_dim=16**3,  # 4096 points, as used in the PointNet++ paper
            augment=True,
            n_augs=20,
            model_contacts=True,
        ),
        data_loader=dict(batch_size=64),
        model=dict(
            n_obj_keypoints=4096,  # 4096 points taken from the object point cloud's target points of the BPS representation
            y_input_keypoints=4096
            + 21
            + 32,  # 1024 points + 21 MANO joints + 32 contact anchors
            y_embed_dim=256,
            object_in_encoder=False,
            skip_connections=True,
            single_modality="noisy_pair",
        ),
        run=dict(
            conditional=True,
            full_choir=False,
            model_contacts=True,  # Must be equal to object_in_encoder!
        ),  # We can reuse the "full_choir" flag for "hand_object_pair"
        bases=(Experiment,),
    ),
    name="baseline_coddpm_3d_multiview_contactopt_noisy_pair",
)
experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model": "kp_coddpm"},
            {"override /dataset": "contactpose"},
            {"override /trainer": "ddpm_baseline_multiview"},
            {"override /tester": "ddpm_baseline_multiview"},
            {"override /training_loss": "diffusion"},
        ],
        dataset=dict(
            perturbation_level=2,
            max_views_per_grasp=1,
            use_contactopt_splits=False,
            use_improved_contactopt_splits=True,
            remap_bps_distances=True,
            use_deltas=False,
            use_bps_grid=True,  # We won't exploit it but then it's a fairer comparison
            bps_dim=16**3,  # 4096 points, as used in the PointNet++ paper
            augment=True,
            n_augs=20,
            model_contacts=True,
        ),
        data_loader=dict(batch_size=64),
        model=dict(
            n_obj_keypoints=4096,  # 4096 points taken from the object point cloud's target points of the BPS representation
            y_input_keypoints=4096,
            y_embed_dim=256,
            object_in_encoder=False,
            skip_connections=True,
            single_modality="object",
        ),
        run=dict(
            conditional=True,
            full_choir=False,
            model_contacts=True,  # Must be equal to object_in_encoder!
        ),  # We can reuse the "full_choir" flag for "hand_object_pair"
        bases=(Experiment,),
    ),
    name="baseline_coddpm_3d_multiview_contactopt_object",
)
experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model": "bps_ddpm"},
            {"override /dataset": "contactpose"},
            {"override /trainer": "ddpm_multiview"},
            {"override /tester": "ddpm_multiview"},
            {"override /training_loss": "diffusion"},
        ],
        dataset=dict(
            perturbation_level=2,
            max_views_per_grasp=1,
            use_contactopt_splits=False,
            use_improved_contactopt_splits=True,
            remap_bps_distances=True,
            use_deltas=False,
            use_bps_grid=True,
            bps_dim=16**3,  # 4096 points
            augment=True,
            n_augs=20,
        ),
        data_loader=dict(batch_size=64),
        model=dict(
            y_embed_dim=256,
            context_channels=MISSING,
            choir_dim=1,
            rescale_input=True,
            backbone="3d_unet",
            embed_full_choir=False,
            use_encoder_self_attn=False,
            use_backbone_self_attn=True,
        ),
        run=dict(conditional=True, full_choir=False),
        bases=(Experiment,),
    ),
    name="cddpm_3d_multiview_contactopt",
)
experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model": "contact_bps_ddpm"},
            {"override /dataset": "contactpose"},
            {"override /trainer": "ddpm_multiview"},
            {"override /tester": "ddpm_multiview"},
            {"override /training_loss": "diffusion"},
        ],
        dataset=dict(
            perturbation_level=2,
            max_views_per_grasp=1,
            use_contactopt_splits=False,
            use_improved_contactopt_splits=True,
            remap_bps_distances=True,
            use_deltas=False,
            use_bps_grid=True,
            bps_dim=16**3,  # 4096 points
            augment=True,
            n_augs=40,
            model_contacts=True,
        ),
        data_loader=dict(batch_size=64),
        model=dict(
            y_embed_dim=128,
            context_channels=MISSING,
            use_encoder_self_attn=False,
            use_backbone_self_attn=True,
            object_in_encoder=True,
            contacts_hidden_dim=1024,
            contacts_skip_connections=True,
            input_normalization="scale",
            conv_transpose=False,
        ),
        run=dict(conditional=True, full_choir=False, model_contacts=True),
        bases=(Experiment,),
    ),
    name="coddpm_3d_multiview_contactopt_multimodal",
)
experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model": "contact_bps_ddpm"},
            {"override /dataset": "contactpose"},
            {"override /trainer": "ddpm_multiview"},
            {"override /tester": "ddpm_multiview"},
            {"override /training_loss": "diffusion"},
        ],
        dataset=dict(
            perturbation_level=2,
            max_views_per_grasp=1,
            use_contactopt_splits=False,
            use_improved_contactopt_splits=True,
            remap_bps_distances=True,
            use_deltas=False,
            use_bps_grid=True,
            bps_dim=16**3,  # 4096 points
            augment=True,
            n_augs=40,
            model_contacts=True,
        ),
        data_loader=dict(batch_size=64),
        model=dict(
            y_embed_dim=128,
            context_channels=MISSING,
            use_encoder_self_attn=False,
            use_backbone_self_attn=True,
            object_in_encoder=True,
            contacts_hidden_dim=1024,
            contacts_skip_connections=True,
            input_normalization="scale",
            single_modality="object",
            conv_transpose=True,
        ),
        run=dict(conditional=True, full_choir=False, model_contacts=True),
        bases=(Experiment,),
    ),
    name="coddpm_3d_multiview_contactopt_object",
)
experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model": "contact_bps_ddpm"},
            {"override /dataset": "contactpose"},
            {"override /trainer": "ddpm_multiview"},
            {"override /tester": "ddpm_multiview"},
            {"override /training_loss": "diffusion"},
        ],
        dataset=dict(
            perturbation_level=2,
            max_views_per_grasp=1,
            use_contactopt_splits=False,
            use_improved_contactopt_splits=True,
            remap_bps_distances=True,
            use_deltas=False,
            use_bps_grid=True,
            bps_dim=16**3,  # 4096 points
            augment=True,
            n_augs=40,
            model_contacts=True,
        ),
        data_loader=dict(batch_size=64),
        model=dict(
            y_embed_dim=128,
            context_channels=MISSING,
            use_encoder_self_attn=False,
            use_backbone_self_attn=True,
            object_in_encoder=True,
            contacts_hidden_dim=1024,
            contacts_skip_connections=True,
            input_normalization="scale",
            single_modality="noisy_pair",
            conv_transpose=True,
        ),
        run=dict(conditional=True, full_choir=False, model_contacts=True),
        bases=(Experiment,),
    ),
    name="coddpm_3d_multiview_contactopt_noisy_pair",
)


experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model": "contact_bps_ddpm"},
            {"override /dataset": "oakink"},
            {"override /trainer": "ddpm_multiview"},
            {"override /tester": "ddpm_multiview"},
            {"override /training_loss": "diffusion"},
        ],
        dataset=dict(
            perturbation_level=2,
            max_views_per_grasp=1,
            remap_bps_distances=True,
            use_deltas=False,
            use_bps_grid=True,
            bps_dim=16**3,  # 4096 points
            augment=True,
            n_augs=5,
            model_contacts=True,
        ),
        data_loader=dict(batch_size=64),
        model=dict(
            y_embed_dim=128,
            context_channels=MISSING,
            use_encoder_self_attn=False,
            use_backbone_self_attn=True,
            object_in_encoder=True,
            contacts_hidden_dim=2048,
            contacts_skip_connections=True,
            input_normalization="scale",
        ),
        run=dict(conditional=True, full_choir=False, model_contacts=True),
        bases=(Experiment,),
    ),
    name="coddpm_3d_multiview_oakink_multimodal",
)

experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model": "contact_bps_ddpm"},
            {"override /dataset": "oakink"},
            {"override /trainer": "ddpm_multiview"},
            {"override /tester": "ddpm_multiview"},
            {"override /training_loss": "diffusion"},
        ],
        dataset=dict(
            perturbation_level=2,
            max_views_per_grasp=1,
            remap_bps_distances=True,
            use_deltas=False,
            use_bps_grid=True,
            bps_dim=16**3,  # 4096 points
            augment=True,
            n_augs=5,
            model_contacts=True,
        ),
        data_loader=dict(batch_size=64),
        model=dict(
            y_embed_dim=128,
            context_channels=MISSING,
            use_encoder_self_attn=False,
            use_backbone_self_attn=True,
            object_in_encoder=True,
            contacts_hidden_dim=2048,
            contacts_skip_connections=True,
            input_normalization="scale",
            single_modality="object",
        ),
        run=dict(conditional=True, full_choir=False, model_contacts=True),
        bases=(Experiment,),
    ),
    name="coddpm_3d_multiview_oakink_object",
)

experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model": "contact_bps_ddpm"},
            {"override /dataset": "oakink"},
            {"override /trainer": "ddpm_multiview"},
            {"override /tester": "ddpm_multiview"},
            {"override /training_loss": "diffusion"},
        ],
        dataset=dict(
            perturbation_level=2,
            max_views_per_grasp=1,
            remap_bps_distances=True,
            use_deltas=False,
            use_bps_grid=True,
            bps_dim=16**3,  # 4096 points
            augment=True,
            n_augs=5,
            model_contacts=True,
        ),
        data_loader=dict(batch_size=64),
        model=dict(
            y_embed_dim=128,
            context_channels=MISSING,
            use_encoder_self_attn=False,
            use_backbone_self_attn=True,
            object_in_encoder=True,
            contacts_hidden_dim=2048,
            contacts_skip_connections=True,
            input_normalization="scale",
            single_modality="noisy_pair",
        ),
        run=dict(conditional=True, full_choir=False, model_contacts=True),
        bases=(Experiment,),
    ),
    name="coddpm_3d_multiview_oakink_noisy_pair",
)

experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model": "contact_bps_ddpm"},
            {"override /dataset": "grab"},
            {"override /trainer": "ddpm_multiview"},
            {"override /tester": "seq_ddpm"},
            {"override /training_loss": "diffusion"},
        ],
        dataset=dict(
            perturbation_level=1,
            min_views_per_grasp=2,
            max_views_per_grasp=15,
            use_affine_mano=True,
            static_grasps_only=False,
            remap_bps_distances=True,
            use_deltas=False,
            use_bps_grid=True,
            bps_dim=16**3,  # 4096 points
            # model_contacts=True,
        ),
        data_loader=dict(batch_size=64),
        model=dict(
            y_embed_dim=128,
            context_channels=MISSING,
            use_encoder_self_attn=False,
            use_backbone_self_attn=True,
            object_in_encoder=True,
            contacts_hidden_dim=2048,
            contacts_skip_connections=True,
            input_normalization="scale",
            single_modality="noisy_pair",
        ),
        run=dict(conditional=True, full_choir=False, model_contacts=True),
        bases=(Experiment,),
    ),
    name="coddpm_3d_multiview_grab_noisy_pair",
)

experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model": "contact_bps_ddpm"},
            {"override /dataset": "grab"},
            {"override /trainer": "ddpm_multiview"},
            {"override /tester": "seq_ddpm"},
            {"override /training_loss": "diffusion"},
        ],
        dataset=dict(
            perturbation_level=1,
            min_views_per_grasp=2,
            max_views_per_grasp=15,
            use_affine_mano=True,
            static_grasps_only=False,
            remap_bps_distances=True,
            use_deltas=False,
            use_bps_grid=True,
            bps_dim=16**3,  # 4096 points
            # model_contacts=True,
        ),
        data_loader=dict(batch_size=64),
        model=dict(
            y_embed_dim=128,
            context_channels=MISSING,
            use_encoder_self_attn=False,
            use_backbone_self_attn=True,
            object_in_encoder=True,
            contacts_hidden_dim=2048,
            contacts_skip_connections=True,
            input_normalization="scale",
        ),
        run=dict(conditional=True, full_choir=False, model_contacts=True),
        bases=(Experiment,),
    ),
    name="coddpm_3d_multiview_grab_multimodal",
)

experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model": "bps_ddpm"},
            {"override /dataset": "contactpose"},
            {"override /trainer": "ddpm_multiview"},
            {"override /tester": "ddpm_multiview"},
            {"override /training_loss": "diffusion"},
        ],
        dataset=dict(
            perturbation_level=2,
            max_views_per_grasp=1,
            use_contactopt_splits=False,
            use_improved_contactopt_splits=True,
            remap_bps_distances=True,
            use_deltas=False,
            use_bps_grid=True,
            bps_dim=16**3,  # 4096 points
            augment=True,
            n_augs=20,
        ),
        data_loader=dict(batch_size=64),
        model=dict(
            y_embed_dim=256,
            context_channels=256,
            choir_dim=1,
            rescale_input=True,
            backbone="3d_unet_w_transformer_spatial_patches",
            embed_full_choir=False,
            use_encoder_self_attn=False,
            use_backbone_self_attn=True,
        ),
        run=dict(conditional=True, full_choir=False),
        bases=(Experiment,),
    ),
    name="cddpm_tr_3d_multiview_contactopt",
)
experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model": "bps_ddpm"},
            {"override /dataset": "contactpose"},
            {"override /trainer": "ddpm"},
            {"override /tester": "ddpm"},
            {"override /training_loss": "diffusion"},
        ],
        dataset=dict(
            perturbation_level=0,
            remap_bps_distances=False,
            use_deltas=True,
            use_bps_grid=True,
            bps_dim=16**3,  # 32**3 should give much better results
        ),
        data_loader=dict(batch_size=64),
        model=dict(
            y_embed_dim=None,
            choir_dim=3,
            rescale_input=False,
            backbone="3d_unet",
        ),
        bases=(Experiment,),
    ),
    name="ddpm_3d_deltas",
)


experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model": "bps_ddpm"},
            {"override /dataset": "contactpose"},
            {"override /trainer": "ddpm"},
            {"override /tester": "ddpm"},
            {"override /training_loss": "diffusion"},
        ],
        dataset=dict(perturbation_level=0),
        data_loader=dict(batch_size=64),
        model=dict(
            y_embed_dim=512,
            choir_dim=1,
            rescale_input=True,
        ),
        run=dict(conditional=True),
        bases=(Experiment,),
    ),
    name="cddpm",
)

experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model": "agg_ved"},
            {"override /trainer": "multiview"},
            {"override /tester": "multiview"},
        ],
        dataset=dict(perturbation_level=2),
        training_loss=dict(multi_view=True),
        data_loader=dict(batch_size=32),
        model=dict(latent_dim=128),
        # model=dict(latent_dim=16, encoder_layer_dims=(4096, 2048, 1024, 512, 256),
        # decoder_layer_dims=(128, 256, 512, 1024)),
        # model=dict(encoder_layer_dims=(1024, 512, 256), decoder_layer_dims=(256, 512),
        bases=(Experiment,),
    ),
    name="multiview",
)


experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model": "agg_ved"},
            {"override /trainer": "multiview"},
            {"override /tester": "multiview"},
        ],
        dataset=dict(
            perturbation_level=2,
            max_views_per_grasp=4,
            use_contactopt_splits=True,
            augment=False,
        ),
        training_loss=dict(multi_view=True),
        data_loader=dict(batch_size=32),
        model=dict(latent_dim=128),
        # model=dict(latent_dim=16, encoder_layer_dims=(4096, 2048, 1024, 512, 256),
        # decoder_layer_dims=(128, 256, 512, 1024)),
        # model=dict(encoder_layer_dims=(1024, 512, 256), decoder_layer_dims=(256, 512),
        bases=(Experiment,),
    ),
    name="multiview_contactopt_replica",
)

experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model": "agg_ved"},
            {"override /trainer": "multiview"},
            {"override /tester": "multiview"},
        ],
        dataset=dict(
            perturbation_level=2,
            max_views_per_grasp=4,
            use_contactopt_splits=False,
            use_improved_contactopt_splits=True,
            augment=True,
            n_augs=20,
        ),
        training_loss=dict(multi_view=True, kl_w=1e-11, use_kl_scheduler=True),
        data_loader=dict(batch_size=64, num_workers=4, prefetch_factor=2),
        model=dict(
            latent_dim=128,
            encoder_layer_dims=(1024, 1024, 1024, 1024, 1024),
            decoder_layer_dims=(1024, 1024, 1024),
            residual_connections=True,
        ),
        bases=(Experiment,),
    ),
    name="multiview_contactopt",
)


experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model": "agg_ved"},
            {"override /trainer": "multiview"},
            {"override /tester": "multiview"},
            {"override /dataset": "grab"},
        ],
        dataset=dict(
            perturbation_level=1,
            min_views_per_grasp=2,
            max_views_per_grasp=15,
            use_affine_mano=True,
            static_grasps_only=False,
        ),
        training_loss=dict(multi_view=True, temporal=True),
        data_loader=dict(batch_size=32),  # , n_batches=100),
        model=dict(
            latent_dim=128,
            # encoder_layer_dims=(1024, 1024, 1024, 1024, 1024),
            # decoder_layer_dims=(2048, 2048, 2048),
            aggregator="attention_pytorch",
        ),
        bases=(Experiment,),
    ),
    name="multiview_grab",
)


experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model": "grasp_tta_cvae"},
            {"override /trainer": "grasp_tta_cvae"},
            {"override /tester": "grasp_tta_cvae"},
            {"override /dataset": "contactpose"},
            {"override /training_loss": "grasp_tta_cvae"},
        ],
        dataset=dict(
            perturbation_level=2,
            max_views_per_grasp=1,
            use_improved_contactopt_splits=True,
            augment=True,
            n_augs=20,
            obj_ptcld_size=3000,
            load_pointclouds=True,
        ),
        model=dict(mano_params_dim=37),
        data_loader=dict(batch_size=128),
        bases=(Experiment,),
    ),
    name="grasp_tta_cvae_contactpose",
)


experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model": "grasp_tta_contactnet"},
            {"override /trainer": "grasp_tta_contactnet"},
            {"override /tester": "grasp_tta"},
            {"override /dataset": "contactpose"},
            {"override /training_loss": "grasp_tta_cvae"},
        ],
        dataset=dict(
            perturbation_level=2,
            max_views_per_grasp=1,
            use_improved_contactopt_splits=True,
            augment=True,
            n_augs=20,
            obj_ptcld_size=3000,
            load_pointclouds=True,
        ),
        data_loader=dict(batch_size=64),
        bases=(Experiment,),
    ),
    name="grasp_tta_contactnet_contactpose",
)


experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model": "grasp_tta_wrapper"},
            {"override /trainer": "grasp_tta_contactnet"},
            {"override /tester": "grasp_tta"},
            {"override /dataset": "contactpose"},
            {"override /training_loss": "grasp_tta_cvae"},
        ],
        dataset=dict(
            perturbation_level=2,
            max_views_per_grasp=1,
            use_improved_contactopt_splits=True,
            obj_ptcld_size=3000,
            augment=True,
            n_augs=20,
            load_pointclouds=True,
        ),
        model=dict(mano_params_dim=37),
        data_loader=dict(batch_size=64),
        bases=(Experiment,),
        run=dict(
            compute_pose_error=False,
            compute_contact_scores=False,
            compute_iv=True,
            compute_sim_displacement=True,
            inference_mode="generation",
        ),
    ),
    name="grasp_tta_wrapped_contactpose",
)
