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
from src.losses.hoi import CHOIRFittingLoss, ContactsFittingLoss
from utils import to_cuda, to_cuda_


@to_cuda
def optimize_pose_pca_from_choir(
    choir: torch.Tensor,
    bps: torch.Tensor,
    anchor_indices: torch.Tensor,
    scalar: torch.Tensor,
    remap_bps_distances: bool,
    is_rhand: bool,
    use_smplx: bool,
    dataset: str,
    contact_gaussians: Optional[torch.Tensor] = None,
    obj_pts: Optional[torch.Tensor] = None,
    obj_normals: Optional[torch.Tensor] = None,
    exponential_map_w: Optional[float] = None,
    loss_thresh: float = 1e-11,
    lr: float = 5e-2,
    max_iterations=8000,
    initial_params=None,
    beta_w: float = 0.05,
    theta_w: float = 0.01,
    choir_w: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    /!\ Important /!\: when computing the CHOIR field, the scalar is applied to the pointclouds
    before the exponential map, if they are used. So we must first apply the inverse exponential
    map, and then the inverse of the scalar.
    """
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
                    num_pca_comps=24,
                    flat_hand_mean=False,
                    batch_size=B,
                )
            )
    if dataset.lower() == "contactpose":
        affine_mano = to_cuda_(
            AffineMANO(15, flat_hand_mean=False, for_contactpose=True)
        )
        ncomps = 15
    elif dataset.lower() == "grab":
        affine_mano = to_cuda_(
            AffineMANO(24, flat_hand_mean=True, for_contactpose=False)
        )
        ncomps = 24
    else:
        raise ValueError(f"Unknown dataset '{dataset}'.")
    if initial_params is None:
        theta = torch.randn((B, ncomps + (3 if not use_smplx else 0))) * 0.01
        beta = torch.randn((B, 10)) * 0.01
        if use_smplx:
            rot = torch.randn((B, 3)) * 0.01  # axis-angle for SMPL-X
        else:
            rot = torch.randn((B, 6)) * 0.01  # 6D for AffineMANO
        trans = torch.randn((B, 3)) * 0.001
    else:
        theta = initial_params["theta"].detach().clone()
        beta = initial_params["beta"].detach().clone()
        rot = initial_params["rot"].detach().clone()
        trans = initial_params["trans"].detach().clone()
        theta_reg = initial_params["theta"].detach().clone()
        # If we have more than 1 observation, we average the parameters:
        with torch.no_grad():
            if len(theta.shape) > 2:
                theta = theta.mean(dim=1)
                theta_reg = theta_reg.mean(dim=1)
            if len(beta.shape) > 2:
                beta = beta.mean(dim=1)
            if len(rot.shape) > 2:
                rot = rot.mean(dim=1)
            if len(trans.shape) > 2:
                trans = trans.mean(dim=1)
        assert (
            theta.shape[0] == B
        ), "compute_pca_from_choir(): batch size mismatch between initial parameters and CHOIR field."
    rot, trans, theta, beta = to_cuda_((rot, trans, theta, beta))
    for p in (rot, trans, theta, beta):
        p.requires_grad = True
    optimizer = torch.optim.Adam([rot, trans, theta, beta], lr=lr)

    if initial_params is not None:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.8)
    proc_bar = tqdm.tqdm(range(max_iterations))

    prev_loss = float("inf")

    # ============== Rescale the CHOIR field to fit the MANO model ==============
    if remap_bps_distances:
        assert exponential_map_w is not None
        choir = -torch.log(choir) / exponential_map_w
    assert (
        choir.shape[0] == scalar.shape[0]
    ), "Batch size mismatch between CHOIR and scalar."
    if len(scalar.shape) == 1:
        choir = (
            choir / scalar[:, None, None]
        )  # CHOIR was computed with scaled up MANO and object pointclouds.
        bps = (
            bps[None, :] / scalar[:, None, None]
        )  # BPS should be scaled down to fit the MANO model in the same scale.
    elif len(scalar.shape) == 2:
        choir = (
            choir / scalar[:, None, :]
        )  # CHOIR was computed with scaled up MANO and object pointclouds.
        bps = (
            bps[None, :] / scalar[:, None, :]
        )  # BPS should be scaled down to fit the MANO model in the same scale.

    choir_loss = CHOIRFittingLoss().to(choir.device)

    plateau_cnt = 0
    pose_regularizer = to_cuda_(torch.tensor(0.0))
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
        if anchors.isnan().any():
            raise ValueError("NaNs in anchors.")
        contact_loss = choir_w * choir_loss(anchors, choir, bps, anchor_indices)
        shape_regularizer = (
            beta_w * torch.norm(beta) ** 2
        )  # Encourage the shape parameters to remain close to 0
        if initial_params is not None:
            pose_regularizer = (
                theta_w * torch.norm(theta - theta_reg) ** 2
            )  # Shape prior

        proc_bar.set_description(
            f"Anchors loss: {contact_loss.item():.10f} / Shape reg: {shape_regularizer.item():.10f} / Pose reg: {pose_regularizer.item():.10f}"
        )
        if (
            torch.abs(prev_loss - contact_loss.detach().type(torch.float32))
            <= loss_thresh
        ):
            plateau_cnt += 1
        else:
            plateau_cnt = 0
        prev_loss = contact_loss.detach().type(torch.float32)
        contact_loss = contact_loss + shape_regularizer + pose_regularizer
        contact_loss.backward()
        optimizer.step()
        scheduler.step()
        if plateau_cnt >= 10:
            break

    if contact_gaussians is None:
        return (
            theta.detach(),
            beta.detach(),
            rot.detach(),
            trans.detach(),
            anchors.detach(),
            verts.detach(),
            joints.detach(),
        )

    plateau_cnt = 0
    iterations = 300
    proc_bar = tqdm.tqdm(range(iterations))
    prev_loss = float("inf")
    optimizer.zero_grad()
    trans_base, rot_base = trans.detach().clone(), rot.detach().clone()
    optimizer = torch.optim.Adam([theta, beta, trans, rot], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    assert obj_pts is not None, "obj_pts must be provided if contact_gaussians is."
    assert (
        obj_normals is not None
    ), "obj_normals must be provided if contact_gaussians is."
    contacts_loss = ContactsFittingLoss(
        verts.detach()[0],
        anchors.detach()[0],
        use_median_filter=False,
        median_filter_len=50,
        update_knn_each_step=True,
    ).to(contact_gaussians.device)
    converged_pose = None
    for i in proc_bar:
        enable_penetration_loss = i > 20  # (i > (0.5 * iterations))
        optimizer.zero_grad()
        if use_smplx:
            output = smplx_model(
                hand_pose=theta, betas=beta, global_orient=rot, transl=trans
            )
            verts, joints = output.vertices, output.joints
        else:
            verts, joints = affine_mano(theta, beta, rot, trans)
        anchors = affine_mano.get_anchors(verts)
        contact_loss, penetration_loss = contacts_loss(
            verts,
            anchors.detach(),
            obj_pts,
            contact_gaussians,
            obj_normals=obj_normals if enable_penetration_loss else None,
        )
        contact_loss = 1000 * contact_loss
        penetration_loss = 500 * penetration_loss
        shape_regularizer = (
            10 * torch.norm(beta) ** 2
        )  # Encourage the shape parameters to remain close to 0
        pose_regularizer = 1e-6 * torch.norm(theta) ** 2
        abs_pose_regularizer = 1e-2 * (torch.norm(trans - trans_base) ** 2) + 1e-3 * (
            torch.norm(rot - rot_base) ** 2
        )
        proc_bar.set_description(
            f"Contacts: {contact_loss.item():.8f} / Penetration: {penetration_loss.item():.8f}  / Shape reg: {shape_regularizer.item():.8f} "
            + f"/ Pose reg: {pose_regularizer.item():.8f} / Abs. pose reg: {abs_pose_regularizer.item():.8f}"
        )
        if (
            torch.abs(prev_loss - contact_loss.detach().type(torch.float32)) <= 1e-4
            and enable_penetration_loss
        ):
            plateau_cnt += 1
        else:
            plateau_cnt = 0
        # loss = contact_loss #+ penetration_loss #+ shape_regularizer + pose_regularizer + abs_pose_regularizer
        # loss = contact_loss + abs_pose_regularizer + shape_regularizer + pose_regularizer
        loss = (
            contact_loss + abs_pose_regularizer + pose_regularizer + shape_regularizer
        )
        if enable_penetration_loss:
            deviation_regularizer = 1e-1 * torch.norm(verts - converged_pose) ** 2
            loss += penetration_loss  # + deviation_regularizer
            # loss = contact_loss + penetration_loss + deviation_regularizer
        else:
            converged_pose = verts.detach().clone()
        # loss = loss + shape_regularizer +  abs_pose_regularizer
        prev_loss = contact_loss.detach().type(torch.float32)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if plateau_cnt >= 10:
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


@to_cuda
def optimize_mesh_from_joints_and_anchors(
    joints_and_anchors: torch.Tensor,
    scalar: torch.Tensor,
    is_rhand: bool,
    use_smplx: bool,
    dataset: str,
    loss_thresh: float = 1e-11,
    lr: float = 5e-2,
    max_iterations=8000,
    initial_params=None,
    beta_w: float = 0.05,
    theta_w: float = 0.01,
    choir_w: float = 1.0,
) -> torch.Tensor:
    B = joints_and_anchors.shape[0]
    affine_mano, smplx_model = None, None
    if use_smplx:
        with redirect_stdout(None):
            smplx_model = to_cuda_(
                smplx.create(
                    model_path=project_conf.SMPLX_MODEL_PATH,
                    model_type="mano",
                    is_rhand=is_rhand,
                    num_pca_comps=24,
                    flat_hand_mean=False,
                    batch_size=B,
                )
            )
    if dataset.lower() == "contactpose":
        affine_mano = to_cuda_(
            AffineMANO(15, flat_hand_mean=False, for_contactpose=True)
        )
        ncomps = 15
    elif dataset.lower() == "grab":
        affine_mano = to_cuda_(
            AffineMANO(24, flat_hand_mean=True, for_contactpose=False)
        )
        ncomps = 24
    else:
        raise ValueError(f"Unknown dataset '{dataset}'.")
    if initial_params is None:
        theta = torch.rand((B, ncomps + (3 if not use_smplx else 0))) * 0.01
        beta = torch.rand((B, 10)) * 0.01
        if use_smplx:
            rot = torch.rand((B, 3)) * 0.01  # axis-angle for SMPL-X
        else:
            rot = torch.rand((B, 6)) * 0.01  # 6D for AffineMANO
        trans = torch.rand((B, 3)) * 0.001
    else:
        theta = initial_params["theta"].detach().clone()
        beta = initial_params["beta"].detach().clone()
        rot = initial_params["rot"].detach().clone()
        trans = initial_params["trans"].detach().clone()
        theta_reg = initial_params["theta"].detach().clone()
        # If we have more than 1 observation, we average the parameters:
        with torch.no_grad():
            if len(theta.shape) > 2:
                theta = theta.mean(dim=1)
                theta_reg = theta_reg.mean(dim=1)
            if len(beta.shape) > 2:
                beta = beta.mean(dim=1)
            if len(rot.shape) > 2:
                rot = rot.mean(dim=1)
            if len(trans.shape) > 2:
                trans = trans.mean(dim=1)
        assert theta.shape[0] == B, (
            "optimize_mesh_from_joints_and_anchors(): batch size mismatch between"
            + f" initial parameters and keypoints. theta.shape: {theta.shape}, B: {B}"
        )
    rot, trans, theta, beta = to_cuda_((rot, trans, theta, beta))
    for p in (rot, trans, theta, beta):
        p.requires_grad = True
    optimizer = torch.optim.Adam([rot, trans, theta, beta], lr=lr)

    if initial_params is not None:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.8)
    proc_bar = tqdm.tqdm(range(max_iterations))

    prev_loss = float("inf")

    # ============== Rescale the joints/anchors to fit the MANO model ==============
    if len(scalar.shape) == 1:
        joints_and_anchors = (
            joints_and_anchors / scalar[:, None, None]
        )  # CHOIR (and so joints and anchors) was computed with scaled up MANO and object pointclouds.
    elif len(scalar.shape) == 2:
        joints_and_anchors = (
            joints_and_anchors / scalar[:, None, :]
        )  # CHOIR was computed with scaled up MANO and object pointclouds.

    plateau_cnt = 0
    pose_regularizer = to_cuda_(torch.tensor(0.0))
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
        if anchors.isnan().any():
            raise ValueError("NaNs in anchors.")
        loss = torch.nn.functional.mse_loss(
            joints_and_anchors, torch.cat((joints, anchors), dim=-2)
        )
        shape_regularizer = (
            beta_w * torch.norm(beta) ** 2
        )  # Encourage the shape parameters to remain close to 0
        if initial_params is not None:
            pose_regularizer = (
                theta_w * torch.norm(theta - theta_reg) ** 2
            )  # Shape prior

        proc_bar.set_description(
            f"Anchors loss: {loss.item():.10f} / Shape reg: {shape_regularizer.item():.10f} / Pose reg: {pose_regularizer.item():.10f}"
        )
        if torch.abs(prev_loss - loss.detach().type(torch.float32)) <= loss_thresh:
            plateau_cnt += 1
        else:
            plateau_cnt = 0
        prev_loss = loss.detach().type(torch.float32)
        loss = loss + shape_regularizer + pose_regularizer
        loss.backward()
        optimizer.step()
        scheduler.step()
        if plateau_cnt >= 10:
            break

    return verts.detach()


def get_dict_from_sample_and_label_tensors(
    sample: torch.Tensor, label: torch.Tensor, theta_dim: int = 18, beta_dim: int = 10
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
    - Theta: (B, T, 1, theta_dim)
    - Beta: (B, T, 1, beta_dim)
    - Rot: (B, T, 1, 6)
    - Trans: (B, T, 1, 3)
    ======= Labels ========
    - CHOIR: (B, T, P, 2)
    - Ref. pts: (B, T, P, 3)
    - Scalar: (B, T, 1, 1)
    - Joints: (B, T, 21, 3)
    - Anchors: (B, T, 32, 3)
    - Theta: (B, T, 1, theta_dim)
    - Beta: (B, T, 1, beta_dim)
    - Rot: (B, T, 1, 6)
    - Trans: (B, T, 1, 3)
    =======================

    Sample: (B, T, 3, P=BPS_DIM, D=3) for 3 padded tensors
    Label: (B, T, 9, P=BPS_DIM, D=18) for 9 padded tensors
    """
    noisy_choir, rescaled_ref_pts, scalar, is_rhand, theta, beta, rot_6d, trans = (
        sample[:, :, 0, :, :2],
        sample[:, :, 1, :, :3],
        sample[:, :, 2, 0, 0].squeeze(),
        sample[:, :, 3, 0, 0].squeeze().bool(),
        sample[:, :, 4, 0, :theta_dim],
        sample[:, :, 5, 0, :beta_dim],
        sample[:, :, 6, 0, :6],
        sample[:, :, 7, 0, :3],
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
        label[:, :, 5, 0, :theta_dim],
        label[:, :, 6, 0, :beta_dim],
        label[:, :, 7, 0, :6],
        label[:, :, 8, 0, :3],
    )
    return {
        "choir": noisy_choir,
        "rescaled_ref_pts": rescaled_ref_pts,
        "scalar": scalar,
        "is_rhand": is_rhand,
        "theta": theta,
        "beta": beta,
        "rot": rot_6d,
        "trans": trans,
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
