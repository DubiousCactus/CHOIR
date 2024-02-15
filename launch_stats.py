#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2024 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Launch the stats computation.
"""


import torch
from hydra_zen.typing import Partial
from tqdm import tqdm

import conf.experiment  # Must import the config to add all components to the store!
from src.base_trainer import BaseTrainer


def launch_stats_computation(
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
    "============ Partials instantiation ============"
    train_dataset, val_dataset, test_dataset = None, None, None
    train_dataset, val_dataset, test_dataset = (
        dataset(split="train", seed=run.seed),
        dataset(split="val", seed=run.seed),
        dataset(split="test", augment=False, seed=run.seed),
    )

    test_dataset.set_observations_number(1)

    train_loader_inst, val_loader_inst, test_loader_inst = None, None, None
    train_loader_inst = data_loader(
        train_dataset, batch_size=1, num_workers=0, persistent_workers=False
    )
    val_loader_inst = data_loader(
        val_dataset,
        shuffle=False,
        drop_last=False,
        n_batches=None,
        batch_size=1,
        num_workers=0,
        persistent_workers=False,
    )
    test_loader_inst = data_loader(
        test_dataset,
        shuffle=False,
        drop_last=False,
        n_batches=None,
        batch_size=1,
        num_workers=0,
        persistent_workers=False,
    )

    gt_udf_vals, gt_gaussian_vals = [], []
    input_udf_vals = []
    gt_kp_vals, input_kp_vals = [], []
    for batch in tqdm(train_loader_inst):
        samples, labels, _ = batch
        label_choir = labels["choir"]
        gt_udf_vals.append(label_choir[0, 0, :, :2])
        gt_gaussian_vals.append(label_choir[0, 0, :, 2:])
        input_udf_vals.append(samples["choir"][0, 0, :, :2])
        gt_kp_vals.append(
            torch.cat(
                (
                    labels["rescaled_ref_pts"][0, 0],
                    labels["joints"][0, 0],
                    labels["anchors"][0, 0],
                ),
                dim=-2,  # Concat along the keypoints and not their dimensionality
            )
        )
        input_kp_vals.append(
            torch.cat(
                (
                    samples["rescaled_ref_pts"][0, 0],
                    samples["joints"][0, 0],
                    samples["anchors"][0, 0],
                ),
                dim=-2,  # Concat along the keypoints and not their dimensionality
            )
        )

    for batch in tqdm(val_loader_inst):
        samples, labels, _ = batch
        label_choir = labels["choir"]
        gt_udf_vals.append(label_choir[0, 0, :, :2])
        gt_gaussian_vals.append(label_choir[0, 0, :, 2:])
        input_udf_vals.append(samples["choir"][0, 0, :, :2])
        gt_kp_vals.append(
            torch.cat(
                (
                    labels["rescaled_ref_pts"][0, 0],
                    labels["joints"][0, 0],
                    labels["anchors"][0, 0],
                ),
                dim=-2,  # Concat along the keypoints and not their dimensionality
            )
        )
        input_kp_vals.append(
            torch.cat(
                (
                    samples["rescaled_ref_pts"][0, 0],
                    samples["joints"][0, 0],
                    samples["anchors"][0, 0],
                ),
                dim=-2,  # Concat along the keypoints and not their dimensionality
            )
        )
    for batch in tqdm(test_loader_inst):
        samples, labels, _ = batch
        label_choir = labels["choir"]
        gt_udf_vals.append(label_choir[0, 0, :, :2])
        gt_gaussian_vals.append(label_choir[0, 0, :, 2:])
        input_udf_vals.append(samples["choir"][0, 0, :, :2])
        gt_kp_vals.append(
            torch.cat(
                (
                    labels["rescaled_ref_pts"][0, 0],
                    labels["joints"][0, 0],
                    labels["anchors"][0, 0],
                ),
                dim=-2,  # Concat along the keypoints and not their dimensionality
            )
        )
        input_kp_vals.append(
            torch.cat(
                (
                    samples["rescaled_ref_pts"][0, 0],
                    samples["joints"][0, 0],
                    samples["anchors"][0, 0],
                ),
                dim=-2,  # Concat along the keypoints and not their dimensionality
            )
        )

    gt_udf_vals = torch.cat(gt_udf_vals, dim=0)
    gt_gaussian_vals = torch.cat(gt_gaussian_vals, dim=0)
    input_udf_vals = torch.cat(input_udf_vals, dim=0)
    gt_kp_vals = torch.cat(gt_kp_vals, dim=0)
    input_kp_vals = torch.cat(input_kp_vals, dim=0)

    gt_udf_mean = gt_udf_vals.mean(dim=0)
    gt_udf_std = gt_udf_vals.std(dim=0)
    gt_gaussian_mean = gt_gaussian_vals.mean(dim=0)
    gt_gaussian_std = gt_gaussian_vals.std(dim=0)
    gt_kp_mean = gt_kp_vals.mean(dim=0)
    gt_kp_std = gt_kp_vals.std(dim=0)
    input_udf_mean = input_udf_vals.mean(dim=0)
    input_udf_std = input_udf_vals.std(dim=0)
    input_kp_mean = input_kp_vals.mean(dim=0)
    input_kp_std = input_kp_vals.std(dim=0)
    print(f"[*] Ground-truth UDF: mean={gt_udf_mean}, std={gt_udf_std}")
    print(
        f"[*] Ground-truth Gaussians : mean={gt_gaussian_mean}, std={gt_gaussian_std}"
    )
    print(f"[*] Input UDF: mean={input_udf_mean}, std={input_udf_std}")
    print(f"[*] Input keypoints: mean={input_kp_mean}, std={input_kp_std}")
    print(f"[*] Ground-truth keypoints: mean={gt_kp_mean}, std={gt_kp_std}")
