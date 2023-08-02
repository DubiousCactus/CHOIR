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

from contextlib import redirect_stdout
from typing import Dict, Optional, Tuple

import smplx
import torch
import tqdm

import conf.project as project_conf
from model.affine_mano import AffineMANO
from src.losses.hoi import CHOIRFittingLoss
from utils import to_cuda_


def optimize_pose_pca_from_choir(
    choir: torch.Tensor,
    bps: torch.Tensor,
    scalar: torch.Tensor,
    remap_bps_distances: bool,
    is_rhand: bool,
    use_smplx: bool,
    exponential_map_w: Optional[float] = None,
    loss_thresh: float = 1e-10,
    lr: float = 5e-2,
    max_iterations=8000,
    initial_params=None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    /!\ Important /!\: when computing the CHOIR field, the scalar is applied to the pointclouds
    before the exponential map, if they are used. So we must first apply the inverse exponential
    map, and then the inverse of the scalar.
    """
    ncomps = 24 if use_smplx else 15  # 24 for GRAB, 15 for ContactPose
    assert len(choir.shape) == 3
    B = choir.shape[0]
    affine_mano, smplx_model = None, None
    if use_smplx:
        with redirect_stdout(None):
            smplx_model = to_cuda_(
                smplx.create(
                    model_path=project_conf.SMPLX_MODEL_PATH,
                    model_type="mano",
                    is_rhand=is_rhand,
                    num_pca_comps=ncomps,
                    flat_hand_mean=False,
                    batch_size=B,
                )
            )
    affine_mano = to_cuda_(AffineMANO(ncomps))
    if initial_params is None:
        theta = (
            (torch.rand((B, ncomps + (3 if not use_smplx else 0))) * 0.01)
            .cuda()
            .requires_grad_(True)
        )
        beta = (torch.rand((B, 10)) * 0.01).cuda().requires_grad_(True)
        if use_smplx:
            rot = (
                (torch.rand((B, 3)) * 0.01).cuda().requires_grad_(True)
            )  # axis-angle for SMPL-X
        else:
            rot = (
                (torch.rand((B, 6)) * 0.01).cuda().requires_grad_(True)
            )  # 6D for AffineMANO
        trans = (torch.rand((B, 3)) * 0.001).cuda().requires_grad_(True)
    else:
        theta = initial_params["pose"].cuda().requires_grad_(True)
        beta = initial_params["shape"].cuda().requires_grad_(True)
        rot = initial_params["rot_6d"].cuda().requires_grad_(True)
        trans = initial_params["trans"].cuda().requires_grad_(True)
    parameters = {
        "rot": rot,
        "trans": trans,
        "fingers_pose": theta,
        "shape": beta,
    }
    params = [{"params": parameters.values()}]

    optimizer = torch.optim.Adam(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.8)
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

    plateau_cnt = 0
    for _ in proc_bar:
        optimizer.zero_grad()
        if use_smplx:
            output = smplx_model(
                hand_pose=theta, betas=beta, global_orient=rot, transl=trans
            )
            verts, joints = output.vertices, output.joints
        else:
            verts, joints = affine_mano(theta, beta, rot, trans)
        anchors = affine_mano.get_anchors(verts)
        loss = choir_loss(anchors, choir, bps)
        regularizer = torch.norm(
            beta
        )  # ** 2  # Encourage the shape parameters to remain close to 0

        proc_bar.set_description(f"Anchors loss: {loss.item():.10f}")
        loss = loss + regularizer
        if torch.abs(prev_loss - loss.detach().type(torch.float64)) < loss_thresh:
            plateau_cnt += 1
        prev_loss = loss.detach().type(torch.float64)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if plateau_cnt >= 5:
            break

    return (
        theta.detach(),
        beta.detach(),
        rot.detach(),
        trans.detach(),
        anchors.detach(),
        verts.detach(),
        joints.detach(),
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
    noisy_choir, rescaled_ref_pts, scalar, is_rhand = (
        sample[:, :, 0, :, :2],
        sample[:, :, 1],
        sample[:, :, 2, 0, 0].squeeze(),
        sample[:, :, 3, 0, 0].squeeze().bool(),
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
        "is_rhand": is_rhand,
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
