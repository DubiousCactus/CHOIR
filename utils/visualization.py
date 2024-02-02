#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Visualization utilities.
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import open3d
import pyvista as pv
import torch
from trimesh import Trimesh

import conf.project as project_conf
from model.affine_mano import AffineMANO
from utils import to_cuda, to_cuda_
from utils.dataset import drop_fingertip_joints, snap_to_original_mano
from utils.training import (
    optimize_mesh_from_joints_and_anchors,
    optimize_pose_pca_from_choir,
)


def add_choir_to_plot(
    plot, bps, choir, ref_pts, anchors, hand_mesh=None, hand_only=False, hand_color=None
):
    if len(choir.shape) == 3:
        choir = choir[0]
    if len(ref_pts.shape) == 3:
        ref_pts = ref_pts[0]
    if len(anchors.shape) == 3:
        anchors = anchors[0]
    plot.add_points(
        ref_pts.cpu().numpy(),
        color="blue",
        name="target_points",
        opacity=0.9,
    )

    if not hand_only:
        min_dist, max_dist = choir[:, 1].min(), choir[:, 1].max()
        max_dist = max_dist + 1e-6
        for i in range(bps.shape[0]):
            # The color is proportional to the distance to the anchor. It is in hex format.
            # It is obtained from min-max normalization of the distance in the choir field,
            # without known range. The color range is 0 to 16777215.
            index = i % 32
            anchor = anchors[index, :]
            color = int((choir[i, 1] - min_dist) / (max_dist - min_dist) * 16777215)
            plot.add_lines(
                np.array(
                    [
                        bps[i, :].cpu().numpy(),
                        anchor.cpu().numpy(),
                    ]
                ),
                width=1,
                color="#" + hex(color)[2:].zfill(6),
                name=f"anchor_ray{i}",
            )
    if hand_mesh is not None:
        plot.add_mesh(
            hand_mesh,
            opacity=0.4,
            name="hand_mesh",
            smooth_shading=True,
            **(dict(color=hand_color) if hand_color is not None else {}),
        )
    for i in range(anchors.shape[0]):
        plot.add_mesh(
            pv.Cube(
                center=anchors[i].cpu().numpy(),
                x_length=3e-3,
                y_length=3e-3,
                z_length=3e-3,
            ),
            color="yellow",
            name=f"anchor{i}",
        )


def visualize_model_predictions_with_multiple_views(
    model: torch.nn.Module,
    batch: Union[Tuple, List, torch.Tensor],
    step: int,
    bps: torch.Tensor,
    anchor_indices: torch.Tensor,
    bps_dim: int,
    remap_bps_distances: bool,
    dataset: str,
    theta_dim: str,
    exponential_map_w: Optional[float] = None,
    temporal: bool = False,
    **kwargs,
) -> None:
    """
    This visualization (and all visualization functions in this file) is a hellf of a mess. I just
    got caught up implementing many different models and I didn't have time to refactor the code.
    Sorry.
    Args:
        model (torch.nn.Module): The model to visualize.
        batch (Union[Tuple, List, torch.Tensor]): The batch to process.
        step (int): The current step.
        bps (torch.Tensor): The basis points set.
        anchor_indices (torch.Tensor): The MANO anchor indices for each basis point.
        bps_dim (int): The dimensionality of the basis points set.
        remap_bps_distances (bool): Whether to remap the basis points set distances to [0, 1].
        dataset (str): The dataset name.
        theta_dim (str): The dimensionality of the MANO pose parameters.
        exponential_map_w (Optional[float], optional): The weight of the exponential map (for remap_bps_distances).
        temporal (bool, optional): Whether the model is temporal or not.
    """
    assert bps_dim == bps.shape[0]
    samples, labels, _ = batch
    is_baseline = kwargs.get("method", "aggved") == "baseline"
    if not project_conf.HEADLESS:
        # ============ Get the first element of the batch ============
        input_scalar = samples["scalar"][0].view(-1, *samples["scalar"].shape[2:])
        choir_gt = labels["choir"][0, -1].unsqueeze(0)
        input_choirs = samples["choir"][0].view(-1, *samples["choir"].shape[2:])
        input_ref_pts = samples["rescaled_ref_pts"][0].view(
            -1, *samples["rescaled_ref_pts"].shape[2:]
        )
        gt_scalar = (
            labels["scalar"][0, -1].unsqueeze(0)
            if len(labels["scalar"].shape) >= 2
            else labels["scalar"][-1].unsqueeze(0)
        )
        # gt_scalar = labels["scalar"][0].view(-1, *labels["scalar"].shape[2:])
        gt_ref_pts = labels["rescaled_ref_pts"][0, -1].unsqueeze(0)
        # =============================================================

        if dataset.lower() == "grab":
            affine_mano = to_cuda_(
                AffineMANO(ncomps=24, flat_hand_mean=True, for_contactpose=False)
            )
        elif dataset.lower() == "contactpose":
            affine_mano = to_cuda_(
                AffineMANO(ncomps=15, flat_hand_mean=False, for_contactpose=True)
            )
        else:
            raise NotImplementedError(f"Unknown dataset: {dataset}")
        faces = affine_mano.faces
        F = faces.cpu().numpy()
        # ===================================================================================
        # ============ Display MANO fitted to all noisy input CHOIR fields ==================
        pl = pv.Plotter(
            shape=(1, input_choirs.shape[0] + 1), border=False, off_screen=False
        )
        sample_choirs = samples["choir"][0].view(-1, *samples["choir"].shape[2:])
        print(f"[*] Visualizing {input_choirs.shape[0]} noisy CHOIR fields.")
        for k, v in samples.items():
            print(f"-> {k}: {v.shape}")
        use_smplx = False  # TODO: for now I'm not interested in using it
        with torch.set_grad_enabled(True):
            (
                _,
                _,
                _,
                _,
                anchors_pred,
                verts_pred,
                joints_pred,
            ) = optimize_pose_pca_from_choir(
                sample_choirs,
                bps=bps,
                anchor_indices=anchor_indices,
                scalar=input_scalar,
                is_rhand=samples["is_rhand"][0],
                # max_iterations=5000,
                max_iterations=1000,
                loss_thresh=1e-6,
                lr=8e-2,
                use_smplx=use_smplx,
                dataset=dataset,
                remap_bps_distances=remap_bps_distances,
                exponential_map_w=exponential_map_w,
                initial_params={
                    k: v[0]
                    for k, v in samples.items()
                    if k
                    in ["theta", ("vtemp" if use_smplx else "beta"), "rot", "trans"]
                },
                beta_w=1e-4,
                theta_w=1e-7,
                choir_w=1000,
                # max_iterations=8000,
                # lr=1e-1,
            )
        for i in range(input_choirs.shape[0]):
            pl.subplot(0, i)
            V = verts_pred[i].cpu().numpy()
            tmesh = Trimesh(V, F)
            hand_mesh = pv.wrap(tmesh)
            add_choir_to_plot(
                pl,
                bps / input_scalar[i],
                input_choirs[i] / input_scalar[i],
                input_ref_pts[i] / input_scalar[i],
                anchors_pred[i],
                hand_mesh,
                hand_only=True,
            )
        # ===================================================================================

        # ================== Optimize MANO on model prediction =========================
        if kwargs.get("method", "aggved") == "baseline":
            conditional = kwargs["conditional"]
            n = 1
            print(
                f"Running with Y={torch.cat( ( samples['rescaled_ref_pts'][0], samples['joints'][0], samples['anchors'][0],), dim=-2,).unsqueeze(0).shape}"
            )
            pred = model.generate(
                n,
                y=torch.cat(
                    (
                        samples["rescaled_ref_pts"][0],
                        samples["joints"][0],
                        samples["anchors"][0],
                    ),
                    dim=-2,
                ).unsqueeze(0)
                if conditional
                else None,
            )
            pred = pred[:, 0]  # TODO: Plot all preds
        elif kwargs.get("method", "aggved") == "ddpm":
            conditional = kwargs["conditional"]
            n = 1
            choir_pred = model.generate(
                n,
                y=samples["choir"][0].unsqueeze(0) if conditional else None,
            )
            choir_pred = choir_pred[:, 0]  # TODO: Plot all preds
        elif kwargs.get("method", "aggved") == "aggved":
            y_hat = model(samples["choir"], use_mean=True)
            choir_pred = y_hat["choir"][0].unsqueeze(0)
        mano_params_gt = {
            "pose": labels["theta"],
            "beta": labels["beta"],
            "rot_6d": labels["rot"],
            "trans": labels["trans"],
        }
        mano_params_gt = {k: v[0, -1].unsqueeze(0) for k, v in mano_params_gt.items()}
        gt_pose, gt_shape, gt_rot_6d, gt_trans = tuple(mano_params_gt.values())
        verts_gt, joints_gt = affine_mano(gt_pose, gt_shape, gt_rot_6d, gt_trans)
        anchors_gt = affine_mano.get_anchors(verts_gt)
        with torch.set_grad_enabled(True):
            print(
                "Mean scalar: ",
                torch.mean(input_scalar).unsqueeze(0).to(input_scalar.device),
            )
            if is_baseline:
                joints_pred, anchors_pred = pred[:, :21], pred[:, 21:]
                verts_pred = optimize_mesh_from_joints_and_anchors(
                    pred,
                    scalar=torch.mean(input_scalar)
                    .unsqueeze(0)
                    .to(input_scalar.device),  # TODO: What should I do here?
                    is_rhand=samples["is_rhand"][0],
                    max_iterations=400,
                    loss_thresh=1e-7,
                    lr=8e-2,
                    dataset=dataset,
                    use_smplx=use_smplx,
                    initial_params={
                        k: v[0, -1].unsqueeze(0)
                        for k, v in samples.items()
                        if k
                        in ["theta", ("vtemp" if use_smplx else "beta"), "rot", "trans"]
                    },
                    beta_w=1e-4,
                    theta_w=1e-8,
                )
            else:
                (
                    _,
                    _,
                    _,
                    _,
                    anchors_pred,
                    verts_pred,
                    joints_pred,
                ) = optimize_pose_pca_from_choir(
                    choir_pred,
                    bps=bps,
                    anchor_indices=anchor_indices,
                    scalar=torch.mean(input_scalar)
                    .unsqueeze(0)
                    .to(input_scalar.device),  # TODO: What should I do here?
                    is_rhand=samples["is_rhand"][0],
                    max_iterations=400,
                    loss_thresh=1e-7,
                    lr=8e-2,
                    # max_iterations=8000,
                    # lr=1e-1,
                    use_smplx=use_smplx,
                    dataset=dataset,
                    remap_bps_distances=remap_bps_distances,
                    exponential_map_w=exponential_map_w,
                    initial_params={
                        k: v[0, -1].unsqueeze(0)
                        for k, v in samples.items()
                        if k
                        in ["theta", ("vtemp" if use_smplx else "beta"), "rot", "trans"]
                    },
                    beta_w=1e-4,
                    theta_w=1e-8,
                    choir_w=1000,
                )
        # ============ Display the ground truth CHOIR field with the GT MANO ================
        V_gt = verts_gt[0].cpu().numpy()
        tmesh_gt = Trimesh(V_gt, F)
        gt_hand_mesh = pv.wrap(tmesh_gt)
        pl.subplot(0, input_choirs.shape[0])
        add_choir_to_plot(
            pl,
            bps / gt_scalar,
            choir_gt / gt_scalar,
            gt_ref_pts / gt_scalar,
            anchors_gt,
            gt_hand_mesh,
            hand_only=True,
            hand_color="red",
        )
        pl.add_title("Noisy input views", font="courier", color="k", font_size=20)
        pl.link_views()
        pl.set_background("white")  # type: ignore
        pl.add_camera_orientation_widget()
        pl.show(interactive=True)

        # ============ Display the optimized MANO on predicted CHOIR field vs GT CHOIR field ================
        # ====== Metrics and qualitative comparison ======
        # === Anchor error ===
        print(
            f"Anchor error (mm): {torch.norm(anchors_pred - anchors_gt.to(anchors_pred.device), dim=2).mean(dim=1).mean(dim=0).item() * 1000:.2f}"
        )
        # === MPJPE ===
        pjpe = torch.linalg.vector_norm(
            joints_gt.to(joints_pred.device) - joints_pred, ord=2, dim=-1
        )  # Per-joint position error (B, N, 21)
        mpjpe = torch.mean(pjpe, dim=-1).item()  # Mean per-joint position error (B, N)
        print(f"MPJPE (mm): {mpjpe * 1000:.2f}")
        root_aligned_pjpe = torch.linalg.vector_norm(
            (joints_gt - joints_gt[:, 0, :]).to(joints_pred.device)
            - (joints_pred - joints_pred[:, 0, :]),
            ord=2,
            dim=-1,
        )  # Per-joint position error (B, N, 21)
        root_aligned_mpjpe = torch.mean(
            root_aligned_pjpe, dim=-1
        ).item()  # Mean per-joint position error (B, N)
        print(f"Root-aligned MPJPE (mm): {root_aligned_mpjpe * 1000:.2f}")
        # ====== MPVPE ======
        # Compute the mean per-vertex position error (MPVPE) between the predicted and ground truth
        # hand meshes.
        pvpe = torch.linalg.vector_norm(
            verts_gt - verts_pred, ord=2, dim=-1
        )  # Per-vertex position error (B, N, 778)
        mpvpe = torch.mean(pvpe, dim=-1).item()  # Mean per-vertex position error (B, N)
        print(f"MPVPE (mm): {mpvpe * 1000:.2f}")

        V = verts_pred[0].cpu().numpy()
        tmesh = Trimesh(V, F)
        # hand_mesh = pv.wrap(tmesh)

        visualize_MANO(
            tmesh, obj_ptcld=input_ref_pts[0] / input_scalar[0], gt_hand=gt_hand_mesh
        )


def visualize_model_predictions(
    model: torch.nn.Module,
    batch: Union[Tuple, List, torch.Tensor],
    step: int,
    bps: torch.Tensor,
    anchor_indices: torch.Tensor,
    bps_dim: int,
    dataset: str,
    **kwargs,
) -> None:
    assert bps_dim == bps.shape[0]
    x, y = batch  # type: ignore
    noisy_choir, rescaled_ref_pts, input_scalar = x
    (
        choir_gt,
        # anchor_deltas,
        gt_rescaled_ref_pts,
        scalar_gt,
        joints_gt,
        anchors_gt,
        pose_gt,
        beta_gt,
        rot_gt,
        trans_gt,
    ) = y
    if not project_conf.HEADLESS:
        pred = model(noisy_choir)
        choir_pred = pred["choir"]
        mano_params_gt = {
            "pose": pose_gt,
            "beta": beta_gt,
            "rot_6d": rot_gt,
            "trans": trans_gt,
        }
        visualize_CHOIR_prediction(
            choir_pred,
            choir_gt,
            # pcl_mean,
            # pcl_scalar,
            bps,
            anchor_indices,
            input_scalar,
            scalar_gt,
            rescaled_ref_pts,
            gt_rescaled_ref_pts,
            mano_params_gt,
            bps_dim=bps_dim,
            dataset=dataset,
        )
    if project_conf.USE_WANDB:
        # TODO: Log a few predictions and the ground truth to wandb.
        # wandb.log({"pointcloud": wandb.Object3D(ptcld)}, step=step)
        # raise NotImplementedError("Visualization is not implemented for wandb.")
        pass


def visualize_ddpm_generation(
    model: torch.nn.Module,
    batch: Union[Tuple, List, torch.Tensor],
    step: int,
    bps: torch.Tensor,
    anchor_indices: torch.Tensor,
    bps_dim: int,
    dataset: str,
    use_deltas: bool,
    conditional: bool,
    multi_view: bool = False,
    **kwargs,
) -> None:
    assert bps_dim == bps.shape[0]
    samples, labels, _ = batch
    if not project_conf.HEADLESS:
        choir_pred = model.generate(
            3,
            y=samples["choir"][0].unsqueeze(0) if conditional else None,
        )
        if use_deltas:
            choir_pred = torch.cat(
                (
                    torch.linalg.norm(choir_pred[..., :3], dim=-1).unsqueeze(-1),
                    torch.linalg.norm(choir_pred[..., 3:], dim=-1).unsqueeze(-1),
                ),
                dim=-1,
            )
        choir_pred = torch.cat((torch.zeros_like(choir_pred), choir_pred), dim=-1)
        mano_params_gt = {
            "pose": labels["theta"],
            "beta": labels["beta"],
            "rot_6d": labels["rot"],
            "trans": labels["trans"],
        }
        if dataset.lower() == "grab":
            affine_mano = to_cuda_(
                AffineMANO(ncomps=24, flat_hand_mean=True, for_contactpose=False)
            )
        elif dataset.lower() == "contactpose":
            affine_mano = to_cuda_(
                AffineMANO(ncomps=15, flat_hand_mean=False, for_contactpose=True)
            )
        else:
            raise NotImplementedError(f"Unknown dataset: {dataset}")
        mano_params_gt = {k: v[0, -1].unsqueeze(0) for k, v in mano_params_gt.items()}
        gt_pose, gt_shape, gt_rot_6d, gt_trans = tuple(mano_params_gt.values())
        # gt_verts, _ = affine_mano(gt_pose, gt_shape, gt_rot_6d, gt_trans)
        gt_verts = to_cuda_(torch.zeros((1, 778, 3)))
        if not multi_view:
            visualize_CHOIR_prediction(
                choir_pred,
                labels["choir"],
                bps,
                anchor_indices,
                samples["scalar"][0].unsqueeze(-1),
                labels["scalar"][0].unsqueeze(-1),
                samples["rescaled_ref_pts"],
                labels["rescaled_ref_pts"],
                gt_verts,
                labels["joints"],
                labels["anchors"],
                is_rhand=samples["is_rhand"][0][0].bool().item(),
                use_smplx=False,
                dataset=dataset,
                remap_bps_distances=kwargs["remap_bps_distances"],
                exponential_map_w=kwargs["exponential_map_w"],
                plot_choir=False,
                use_deltas=use_deltas,
            )
        else:
            pass
    if project_conf.USE_WANDB:
        # TODO: Log a few predictions and the ground truth to wandb.
        # wandb.log({"pointcloud": wandb.Object3D(ptcld)}, step=step)
        # raise NotImplementedError("Visualization is not implemented for wandb.")
        pass


# TODO: Refactor the 2 following functions. It's too WET!!!
@to_cuda
def visualize_CHOIR_prediction(
    choir_pred: torch.Tensor,
    choir_gt: torch.Tensor,
    bps: torch.Tensor,
    anchor_indices: torch.Tensor,
    input_scalar: torch.Tensor,
    gt_scalar: torch.Tensor,
    input_ref_pts: torch.Tensor,
    gt_ref_pts: torch.Tensor,
    gt_verts: torch.Tensor,
    gt_joints: torch.Tensor,
    gt_anchors: torch.Tensor,
    is_rhand: bool,
    use_smplx: bool,
    dataset: str,
    remap_bps_distances: bool,
    exponential_map_w: Optional[float] = None,
    plot_choir: bool = True,
    use_deltas: bool = False,
):
    # ============ Get the first element of the batch ============
    if len(choir_pred.shape) > 2:
        choir_pred = choir_pred[0].unsqueeze(0)
    if len(choir_gt.shape) > 2:
        choir_gt = choir_gt[0].unsqueeze(0)
    if len(bps.shape) > 2:
        bps = bps[0].unsqueeze(0)
    if len(input_scalar.shape) > 2:
        input_scalar = input_scalar[0].unsqueeze(0)
    if len(input_ref_pts.shape) > 2:
        input_ref_pts = input_ref_pts[0].unsqueeze(0)
    if len(gt_scalar.shape) > 2:
        gt_scalar = gt_scalar[0].unsqueeze(0)
    if len(gt_ref_pts.shape) > 2:
        gt_ref_pts = gt_ref_pts[0].unsqueeze(0)
    if len(gt_joints.shape) > 2:
        gt_joints = gt_joints[0].unsqueeze(0)
    if len(gt_anchors.shape) > 2:
        gt_anchors = gt_anchors[0].unsqueeze(0)
    # =============================================================
    if use_deltas:
        # TODO: directly plot the delta vectors?
        print("Converting deltas to distances...")
        _choir_pred = torch.cat(
            (
                torch.linalg.norm(choir_pred[..., :3], dim=-1).unsqueeze(-1),
                torch.linalg.norm(choir_pred[..., 3:], dim=-1).unsqueeze(-1),
            ),
            dim=-1,
        )
        _choir_gt = torch.cat(
            (
                torch.linalg.norm(choir_gt[..., :3], dim=-1).unsqueeze(-1),
                torch.linalg.norm(choir_gt[..., 3:], dim=-1).unsqueeze(-1),
            ),
            dim=-1,
        )
    else:
        _choir_pred = choir_pred
        _choir_gt = choir_gt

    affine_mano = to_cuda_(AffineMANO())
    faces = affine_mano.faces
    F = faces.cpu().numpy()

    # ============ Optimize MANO on the predicted CHOIR field to visualize it ============
    with torch.set_grad_enabled(True):
        (
            pose,
            shape,
            rot_6d,
            trans,
            anchors_pred,
            verts_pred,
            joints_pred,
        ) = optimize_pose_pca_from_choir(
            _choir_pred,
            bps=bps,
            anchor_indices=anchor_indices.int(),
            scalar=input_scalar,
            is_rhand=is_rhand,
            use_smplx=use_smplx,
            dataset=dataset,
            remap_bps_distances=remap_bps_distances,
            exponential_map_w=exponential_map_w,
            max_iterations=1000,
            loss_thresh=1e-6,
            lr=8e-2,
            beta_w=1e-1,
            theta_w=1e-7,
            choir_w=1000,
        )
    # ====== Metrics and qualitative comparison ======
    # === Anchor error ===
    print(
        f"Anchor error (mm): {torch.norm(anchors_pred - gt_anchors.to(anchors_pred.device), dim=2).mean(dim=1).mean(dim=0).item() * 1000:.2f}"
    )
    # === MPJPE ===
    if gt_joints.shape[1] == 16 and joints_pred.shape[1] == 21:
        # SMPL-X ground-truth but MANO prediction.
        # This was obtained with SMPL-X which doesn't include fingertips, so we'll drop them for
        # the prediction if we used MANO.
        # Reorder the joints to match the SMPL-X order
        # (https://github.com/lixiny/manotorch/blob/5738d327a343e7533ad60da64d1629cedb5ae9e7/manotorch/manolayer.py#L240):
        joints_pred = snap_to_original_mano(joints_pred)
        joints_pred = drop_fingertip_joints(joints_pred, definition="mano")
    elif gt_joints.shape[1] == 21 and joints_pred.shape[1] == 16:
        # MANO ground-truth but SMPL-X prediction.
        gt_joints = snap_to_original_mano(gt_joints)
        gt_joints = drop_fingertip_joints(gt_joints, definition="mano")

    pjpe = torch.linalg.vector_norm(
        gt_joints - joints_pred, ord=2, dim=-1
    )  # Per-joint position error (B, N, 21)
    mpjpe = torch.mean(pjpe, dim=-1).item()  # Mean per-joint position error (B, N)
    print(f"MPJPE (mm): {mpjpe * 1000:.2f}")
    root_aligned_pjpe = torch.linalg.vector_norm(
        (gt_joints - gt_joints[:, 0, :]) - (joints_pred - joints_pred[:, 0, :]),
        ord=2,
        dim=-1,
    )  # Per-joint position error (B, N, 21)
    root_aligned_mpjpe = torch.mean(
        root_aligned_pjpe, dim=-1
    ).item()  # Mean per-joint position error (B, N)
    print(f"Root-aligned MPJPE (mm): {root_aligned_mpjpe * 1000:.2f}")
    # ====== MPVPE ======
    # Compute the mean per-vertex position error (MPVPE) between the predicted and ground truth
    # hand meshes.
    pvpe = torch.linalg.vector_norm(
        gt_verts - verts_pred, ord=2, dim=-1
    )  # Per-vertex position error (B, N, 778)
    mpvpe = torch.mean(pvpe, dim=-1).item()  # Mean per-vertex position error (B, N)
    print(f"MPVPE (mm): {mpvpe * 1000:.2f}")

    V = verts_pred[0].cpu().numpy()
    tmesh = Trimesh(V, F)
    hand_mesh = pv.wrap(tmesh)

    V_gt = gt_verts[0].cpu().numpy()
    tmesh_gt = Trimesh(V_gt, F)
    gt_hand_mesh = pv.wrap(tmesh_gt)
    visualize_MANO(
        tmesh, obj_ptcld=input_ref_pts[0] / input_scalar[0], gt_hand=gt_hand_mesh
    )
    # ===================================================================================
    if plot_choir:
        pl = pv.Plotter(shape=(1, 2), border=False, off_screen=False)
        pl.subplot(0, 0)
        add_choir_to_plot(
            pl,
            bps / input_scalar,
            _choir_pred / input_scalar,
            input_ref_pts / input_scalar,
            anchors_pred,
            hand_mesh,
        )
        # ===================================================================================
        # ============ Display the ground truth CHOIR field with the GT MANO ================
        pl.subplot(0, 1)
        add_choir_to_plot(
            pl,
            bps / gt_scalar,
            _choir_gt / gt_scalar,
            gt_ref_pts / gt_scalar,
            gt_anchors,
            gt_hand_mesh,
        )
        pl.link_views()
        pl.set_background("white")  # type: ignore
        pl.add_camera_orientation_widget()
        pl.show(interactive=True)


def visualize_CHOIR(
    choir: torch.Tensor,
    bps: torch.Tensor,
    anchor_indices: torch.Tensor,
    scalar: float,
    # dense_contacts: torch.Tensor,
    verts: torch.Tensor,
    anchors: torch.Tensor,
    # anchor_orientations: torch.Tensor,
    obj_mesh,
    obj_pointcloud,
    reference_obj_points: torch.Tensor,
    affine_mano: AffineMANO,
    use_deltas: bool = False,
):
    assert len(anchors.shape) == 2
    assert len(choir.shape) == 2
    assert len(scalar.shape) == 1
    assert len(verts.shape) == 2
    if torch.is_tensor(scalar):
        scalar = scalar.item()

    if use_deltas:
        # TODO: directly plot the delta vectors?
        _choir = torch.cat(
            (
                torch.linalg.norm(choir[..., :3], dim=-1).unsqueeze(-1),
                torch.linalg.norm(choir[..., 3:], dim=-1).unsqueeze(-1),
            ),
            dim=-1,
        )
    else:
        _choir = choir
    n_anchors = anchors.shape[0]
    faces = affine_mano.faces
    V = scalar * verts.cpu().numpy()
    F = faces.cpu().numpy()
    tmesh = Trimesh(V, F)
    hand_mesh = pv.wrap(tmesh)

    pl = pv.Plotter(shape=(2, 2), border=False, off_screen=False)
    pl.subplot(0, 0)
    pl.add_mesh(
        hand_mesh,
        opacity=0.4,
        name="hand_mesh",
        smooth_shading=True,
        # scalars=dense_contacts.cpu().numpy(),
        # cmap="jet",
    )
    obj_tmesh = Trimesh(obj_mesh.vertices, obj_mesh.triangles)
    obj_tmesh.apply_scale(scalar)
    obj_mesh_pv = pv.wrap(obj_tmesh)
    pl.add_mesh(
        obj_mesh_pv, opacity=0.3, name="obj_mesh", smooth_shading=True, color="red"
    )
    for i in range(n_anchors):
        pl.add_mesh(
            pv.Cube(
                center=scalar * anchors[i].cpu().numpy(),
                x_length=5e-3,
                y_length=5e-3,
                z_length=5e-3,
            ),
            color="yellow",
            name=f"anchor{i}",
        )
    # pl.add_axes_at_origin(line_width=1, labels_off=True)

    # Now we're gonna plot the CHOIR field. For each point in the target point cloud, we'll plot a
    # ray from the point to the closest anchor. The ray will be colored according to the distance
    # to the anchor.
    pl.subplot(0, 1)
    pl.add_points(
        reference_obj_points.cpu().numpy(),
        color="blue",
        name="target_points",
        opacity=0.9,
    )
    pl.add_points(
        scalar * obj_pointcloud.cpu().numpy(),
        color="red",
        name="base_ptcld",
        opacity=0.2,
    )
    pl.add_points(bps.cpu().numpy(), color="green", name="basis_points", opacity=0.9)
    assert len(bps.shape) == len(reference_obj_points.shape)
    rescaled_anchors = anchors * scalar
    for i in range(bps.shape[0]):
        # index = i % n_anchors
        index = anchor_indices[i]
        anchor = rescaled_anchors[index, :]
        # The color is proportional to the distance to the anchor. It is in hex format.
        # It is obtained from min-max normalization of the distance in the choir field,
        # without known range. The color range is 0 to 16777215.
        obj_color = int(
            (_choir[i, 0] - _choir[:, 0].min())
            / (_choir[:, 0].max() - _choir[:, 0].min())
            * 16777215
        )
        anchor_color = int(
            (_choir[i, 1] - _choir[:, 1].min())
            / (_choir[:, 1].max() - _choir[:, 1].min())
            * 16777215
        )
        # This is to check that the delta vectors are correct: (should be the same visual
        # result as above)
        # assert np.allclose(
        # np.array([reference_obj_points[i, :].cpu().numpy(), anchor.cpu().numpy()]),
        # np.array(
        # [
        # reference_obj_points[i, :].cpu().numpy(),
        # (
        # reference_obj_points[i, :].cpu()
        # + (_choir[i, 4] * anchor_orientations[i, :])
        # ).numpy(),
        # ]
        # ),
        # )
        pl.add_lines(
            np.array(
                [bps[i, :].cpu().numpy(), reference_obj_points[i, :].cpu().numpy()]
            ),
            width=1,
            color="#" + hex(obj_color)[2:].zfill(6),
            name=f"obj_ray{i}",
        )
        pl.add_lines(
            np.array([bps[i, :].cpu().numpy(), anchor.cpu().numpy()]),
            width=1,
            color="#" + hex(anchor_color)[2:].zfill(6),
            name=f"anchor_ray{i}",
        )

    for i in range(n_anchors):
        pl.add_mesh(
            pv.Cube(
                center=rescaled_anchors[i].cpu().numpy(),
                x_length=5e-3,
                y_length=5e-3,
                z_length=5e-3,
            ),
            color="yellow",
            name=f"anchor{i}",
        )
    pl.subplot(1, 0)
    pl.add_points(
        reference_obj_points.cpu().numpy(),
        color="blue",
        name="target_points",
    )
    pl.add_points(
        scalar * obj_pointcloud.cpu().numpy(),
        color="red",
        name="base_ptcld",
        opacity=0.2,
    )
    pl.add_points(bps.cpu().numpy(), color="green", name="basis_points", opacity=0.9)
    assert len(bps.shape) == len(reference_obj_points.shape)
    rescaled_anchors = anchors * scalar
    for i in range(bps.shape[0]):
        # index = i % n_anchors
        index = anchor_indices[i]
        anchor = rescaled_anchors[index, :]
        # The color is proportional to the distance to the anchor. It is in hex format.
        # It is obtained from min-max normalization of the distance in the choir field,
        # without known range. The color range is 0 to 16777215.
        obj_color = int(
            (_choir[i, 0] - _choir[:, 0].min())
            / (_choir[:, 0].max() - _choir[:, 0].min())
            * 16777215
        )
        pl.add_lines(
            np.array(
                [bps[i, :].cpu().numpy(), reference_obj_points[i, :].cpu().numpy()]
            ),
            width=1,
            color="#" + hex(obj_color)[2:].zfill(6),
            name=f"obj_ray{i}",
        )
    pl.subplot(1, 1)
    pl.add_points(bps.cpu().numpy(), color="green", name="basis_points", opacity=0.9)
    assert len(bps.shape) == len(reference_obj_points.shape)
    rescaled_anchors = anchors * scalar
    for i in range(bps.shape[0]):
        # index = i % n_anchors
        index = anchor_indices[i]
        anchor = rescaled_anchors[index, :]
        anchor_color = int(
            (_choir[i, 1] - _choir[:, 1].min())
            / (_choir[:, 1].max() - _choir[:, 1].min())
            * 16777215
        )
        pl.add_lines(
            np.array([bps[i, :].cpu().numpy(), anchor.cpu().numpy()]),
            width=1,
            color="#" + hex(anchor_color)[2:].zfill(6),
            name=f"anchor_ray{i}",
        )

    for i in range(n_anchors):
        pl.add_mesh(
            pv.Cube(
                center=rescaled_anchors[i].cpu().numpy(),
                x_length=5e-3,
                y_length=5e-3,
                z_length=5e-3,
            ),
            color="yellow",
            name=f"anchor{i}",
        )
    pl.add_mesh(
        hand_mesh,
        opacity=0.4,
        name="hand_mesh",
        smooth_shading=True,
    )

    pl.link_views()
    pl.set_background("white")  # type: ignore
    pl.add_camera_orientation_widget()
    pl.show(interactive=True)


def visualize_MANO(
    pred_hand: Trimesh,
    obj_mesh: Optional[open3d.geometry.TriangleMesh] = None,
    obj_ptcld: Optional[torch.Tensor] = None,
    gt_hand: Optional[Any] = None,
    save_as: Optional[str] = None,
    opacity: float = 0.4,
    return_cam_pose: bool = False,
    cam_pose: Optional[tuple] = None,
):
    pl = pv.Plotter(off_screen=False)
    if not isinstance(pred_hand, Trimesh):
        pred_hand = Trimesh(pred_hand.vertices, pred_hand.triangles)
    hand_mesh = pv.wrap(pred_hand)
    pl.add_mesh(
        hand_mesh,
        opacity=opacity,
        name="hand_mesh",
        label="Predicted Hand",
        smooth_shading=True,
    )
    if gt_hand is not None:
        if isinstance(gt_hand, Trimesh):
            gt_hand = pv.wrap(gt_hand)
        pl.add_mesh(
            gt_hand,
            opacity=0.2,
            name="gt_hand",
            label="Ground-truth Hand",
            smooth_shading=True,
            color="blue",
        )
    if obj_mesh is not None:
        if not isinstance(obj_mesh, Trimesh):
            obj_mesh = Trimesh(obj_mesh.vertices, obj_mesh.triangles)
        obj_mesh_pv = pv.wrap(obj_mesh)
        pl.add_mesh(
            obj_mesh_pv,
            opacity=opacity,
            name="obj_mesh",
            label="Object mesh",
            smooth_shading=True,
            color="red",
        )
    elif obj_ptcld is not None:
        pl.add_points(
            obj_ptcld.detach().cpu().numpy(),
            color="red",
            name="obj_ptcld",
            label="Object pointcloud",
            opacity=0.2,
        )
    pl.add_title("Fitted MANO vs ground-truth MANO", font_size=30)
    pl.set_background("white")  # type: ignore
    # pl.add_camera_orientation_widget()
    pl.add_legend(loc="upper left", size=(0.1, 0.1))
    if cam_pose is not None:
        pl.camera.model_transform_matrix = cam_pose[1]
        pl.camera.position = cam_pose[0]
    if save_as is not None:
        # pl.show()
        # pl.screenshot(save_as)
        # pl.save_graphic(save_as)
        pl.export_html(save_as)
    else:
        pl.add_axes_at_origin()
        pl.show(interactive=True)
    if return_cam_pose:
        return (pl.camera.position, pl.camera.model_transform_matrix)


class ScenePicAnim:
    def __init__(
        self,
        width=1600,
        height=1600,
    ):
        super().__init__()
        try:
            import scenepic as sp
        except:
            raise Exception(
                "scenepic not installed. Some visualization functions will not work. (I know it's not available on Apple Silicon :("
            )
        self.scene = sp.Scene()
        self.n_frames = 0
        self.main = self.scene.create_canvas_3d(width=width, height=height)
        self.colors = sp.Colors

    def meshes_to_sp(self, meshes: Dict[str, Trimesh]):
        sp_meshes = []
        for mesh_name, mesh in meshes.items():
            params = {
                "vertices": mesh.vertices.astype(np.float32),
                # "normals": mesh.vertex_normals.astype(np.float32),
                "triangles": mesh.faces.astype(np.int32),
                "colors": mesh.visual.vertex_colors.astype(np.float32)[..., :3] / 255.0,
            }
            # params = {'vertices' : m.v.astype(np.float32), 'triangles' : m.f, 'colors' : m.vc.astype(np.float32)}
            # sp_m = sp.Mesh()
            sp_m = self.scene.create_mesh(layer_id=mesh_name)
            sp_m.add_mesh_without_normals(**params)
            if mesh_name == "ground_mesh":
                sp_m.double_sided = True
            sp_meshes.append(sp_m)
        return sp_meshes

    def add_frame(self, meshes: Dict[str, Trimesh]):
        meshes_list = self.meshes_to_sp(meshes)
        if not hasattr(self, "focus_point"):
            self.focus_point = list(meshes.values())[0].centroid
            # center = self.focus_point
            # center[2] = 4
            # rotation = sp.Transforms.rotation_about_z(0)
            # self.camera = sp.Camera(center=center, rotation=rotation, fov_y_degrees=30.0)

        main_frame = self.main.create_frame(focus_point=self.focus_point)
        for i, m in enumerate(meshes_list):
            # self.main.set_layer_settings({layer_names[i]:{}})
            main_frame.add_mesh(m)
        self.n_frames += 1

    def save_animation(self, sp_anim_name):
        self.scene.link_canvas_events(self.main)
        self.scene.save_as_html(sp_anim_name, title=os.path.basename(sp_anim_name))


def visualize_3D_gaussians_on_hand_mesh(
    hand_verts: torch.Tensor,
    hand_faces: torch.Tensor,
    gaussian_params: torch.Tensor,
    base_unit: float,
    hand_color: str = "blue",
):
    geometries = []
    print(gaussian_params.shape)
    for i in range(gaussian_params.shape[0]):
        print(f"Visualizing Gaussian {i + 1} / {gaussian_params.shape[0]}")
        print(gaussian_params[i].shape)
        mean = gaussian_params[i, :3]
        covariance = gaussian_params[i, 3:].reshape(3, 3)
        # Sample points from the Gaussian
        num_points = 5000
        points = np.random.multivariate_normal(mean, covariance, num_points)

        # Create a point cloud from the sampled points
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(points)
        geometries.append(pcd)

    # Create the hand mesh
    hand_mesh = open3d.geometry.TriangleMesh()
    hand_mesh.vertices = open3d.utility.Vector3dVector(hand_verts)
    hand_mesh.triangles = open3d.utility.Vector3iVector(hand_faces)
    if hand_color == "blue":
        color = np.array([0, 0, 255])
    elif hand_color == "red":
        color = np.array([255, 0, 0])
    elif hand_color == "green":
        color = np.array([0, 255, 0])
    elif hand_color == "grey":
        color = np.array([169, 169, 169])
    elif hand_color == "rainbow":
        color = np.random.rand(3)
    hand_mesh.paint_uniform_color(color)

    # Visualize the point cloud
    open3d.visualization.draw_geometries(geometries + [hand_mesh])
