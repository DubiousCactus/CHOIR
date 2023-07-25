#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Training utilities. This is a good place for your code that is used in training (i.e. custom loss
function, visualization code, etc.)
"""

from typing import Dict, Optional, Tuple

import torch
import tqdm

from model.affine_mano import AffineMANO
from src.losses.hoi import CHOIRFittingLoss


def optimize_pose_pca_from_choir(
    choir: torch.Tensor,
    bps: torch.Tensor,
    scalar: torch.Tensor,
    remap_bps_distances: bool,
    exponential_map_w: Optional[float] = None,
    loss_thresh: float = 1e-12,
    max_iterations=8000,
    initial_params=None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    /!\ Important /!\: when computing the CHOIR field, the scalar is applied to the pointclouds
    before the exponential map, if they are used. So we must first apply the inverse exponential
    map, and then the inverse of the scalar.
    """
    ncomps = 15
    affine_mano = AffineMANO(ncomps).cuda()
    B = choir.shape[0]
    if initial_params is None:
        fingers_pose = (torch.rand((B, ncomps + 3))).cuda().requires_grad_(True)
        shape = (torch.rand((B, 10))).cuda().requires_grad_(True)
        rot_6d = torch.rand((B, 6)).cuda().requires_grad_(True)
        trans = (torch.rand((B, 3)) * 0.001).cuda().requires_grad_(True)
    else:
        fingers_pose = initial_params["pose"].cuda().requires_grad_(True)
        shape = initial_params["shape"].cuda().requires_grad_(True)
        rot_6d = initial_params["rot_6d"].cuda().requires_grad_(True)
        trans = initial_params["trans"].cuda().requires_grad_(True)
    parameters = {
        "rot": rot_6d,
        "trans": trans,
        "fingers_pose": fingers_pose,
        "shape": shape,
    }
    params = [{"params": parameters.values()}]

    optimizer = torch.optim.Adam(params, lr=5e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.9)
    proc_bar = tqdm.tqdm(range(max_iterations))

    prev_loss = float("inf")

    # ============== Rescale the CHOIR field to fit the MANO model ==============
    if remap_bps_distances:
        assert exponential_map_w is not None
        choir = -torch.log(choir) / exponential_map_w
    choir = (
        choir / scalar[:, None, None]
    )  # CHOIR was computed with scaled up MANO and object pointclouds.
    bps = (
        bps[None, :] / scalar[:, None, None]
    )  # BPS should be scaled down to fit the MANO model in the same scale.

    choir_loss = CHOIRFittingLoss().to(choir.device)

    for _ in proc_bar:
        optimizer.zero_grad()
        verts, _ = affine_mano(fingers_pose, shape, rot_6d, trans)
        anchors = affine_mano.get_anchors(verts)
        loss = choir_loss(anchors, choir, bps)
        regularizer = (
            torch.norm(shape) ** 2
        )  # Encourage the shape parameters to remain close to 0

        proc_bar.set_description(f"Anchors loss: {loss.item():.10f}")
        loss = loss + regularizer
        if torch.abs(prev_loss - loss.detach()) < loss_thresh:
            break
        prev_loss = loss.detach()
        loss.backward()
        optimizer.step()
        scheduler.step()

    return (
        fingers_pose.detach(),
        shape.detach(),
        rot_6d.detach(),
        trans.detach(),
        anchors.detach(),
    )


def get_dict_from_sample_and_label_tensors(
    sample: torch.Tensor, label: torch.Tensor
) -> Tuple[Dict]:
    """
    These are 0-padded tensors, so we only keep the actual data.
    B: Batch size
    T: Number of frames
    P: Number of points
    D: Dimension of the points
    ======= Samples =======
    - CHOIR: (B, T, P, 2)
    - Ref. pts: (B, T, P, 3)
    - Scalar: (B, T, 1, 1)
    ======= Labels ========
    - CHOIR: (B, T, P, 2)
    - Ref. pts: (B, T, P, 3)
    - Scalar: (B, T, 1, 1)
    - Joints: (B, T, 21, 3)
    - Anchors: (B, T, 32, 3)
    - Theta: (B, T, 1, 18)
    - Beta: (B, T, 1, 10)
    - Rot: (B, T, 1, 6)
    - Trans: (B, T, 1, 3)
    =======================

    Sample: (B, T, 3, P=BPS_DIM, D=3) for 3 padded tensors
    Label: (B, T, 9, P=BPS_DIM, D=18) for 9 padded tensors
    """
    noisy_choir, rescaled_ref_pts, scalar = (
        sample[:, :, 0, :, :2],
        sample[:, :, 1],
        sample[:, :, 2, 0, 0].squeeze(),
    )
    (
        gt_choir,
        gt_ref_pts,
        gt_scalar,
        gt_joints,
        gt_anchors,
        gt_theta,
        gt_beta,
        gt_rot,
        gt_trans,
    ) = (
        label[:, :, 0, :, :2],
        label[:, :, 1, :, :3],
        label[:, :, 2, 0, 0].squeeze(),
        label[:, :, 3, :21, :3],
        label[:, :, 4, :32, :3],
        label[:, :, 5, 0, :18],
        label[:, :, 6, 0, :10],
        label[:, :, 7, 0, :6],
        label[:, :, 8, 0, :3],
    )
    return {
        "choir": noisy_choir,
        "rescaled_ref_pts": rescaled_ref_pts,
        "scalar": scalar,
    }, {
        "choir": gt_choir,
        "rescaled_ref_pts": gt_ref_pts,
        "scalar": gt_scalar,
        "joints": gt_joints,
        "anchors": gt_anchors,
        "theta": gt_theta,
        "beta": gt_beta,
        "rot": gt_rot,
        "trans": gt_trans,
    }
