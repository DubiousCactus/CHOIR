#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Visualization utilities.
"""

from typing import Any, List, Optional, Tuple, Union

import numpy as np
import open3d
import pyvista as pv
import torch
from trimesh import Trimesh

import conf.project as project_conf
from model.affine_mano import AffineMANO
from utils import to_cuda
from utils.training import (
    get_dict_from_sample_and_label_tensors,
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
                x_length=1e-2,
                y_length=1e-2,
                z_length=1e-2,
            ),
            color="yellow",
            name=f"anchor{i}",
        )


def visualize_model_predictions_with_multiple_views(
    model: torch.nn.Module,
    batch: Union[Tuple, List, torch.Tensor],
    step: int,
    bps: torch.Tensor,
    bps_dim: int,
    remap_bps_distances: bool,
    exponential_map_w: Optional[float] = None,
    **kwargs,
) -> None:
    assert bps_dim == bps.shape[0]
    x, y = batch  # type: ignore
    samples, labels = get_dict_from_sample_and_label_tensors(x, y)
    if not project_conf.HEADLESS:
        # ============ Get the first element of the batch ============
        input_scalar = samples["scalar"][0].view(-1, *samples["scalar"].shape[2:])
        choir_gt = labels["choir"][0, 0].unsqueeze(0)
        input_choirs = samples["choir"][0].view(-1, *samples["choir"].shape[2:])
        input_ref_pts = samples["rescaled_ref_pts"][0].view(
            -1, *samples["rescaled_ref_pts"].shape[2:]
        )
        gt_scalar = (
            labels["scalar"][0, 0].unsqueeze(0)
            if len(labels["scalar"].shape) >= 2
            else labels["scalar"][0].unsqueeze(0)
        )
        # gt_scalar = labels["scalar"][0].view(-1, *labels["scalar"].shape[2:])
        gt_ref_pts = labels["rescaled_ref_pts"][0, 0].unsqueeze(0)
        # =============================================================

        affine_mano = AffineMANO().cuda()
        faces = affine_mano.faces
        F = faces.cpu().numpy()
        # ===================================================================================
        # ============ Display MANO fitted to all noisy input CHOIR fields ==================
        pl = pv.Plotter(
            shape=(1, input_choirs.shape[0] + 1), border=False, off_screen=False
        )
        sample_choirs = samples["choir"][0].view(-1, *samples["choir"].shape[2:])
        print(f"[*] Visualizing {input_choirs.shape[0]} noisy CHOIR fields.")
        with torch.set_grad_enabled(True):
            pose, shape, rot_6d, trans, anchors_pred = optimize_pose_pca_from_choir(
                sample_choirs,
                bps=bps,
                scalar=input_scalar,
                is_rhand=samples["is_rhand"][0],
                max_iterations=5000,
                remap_bps_distances=remap_bps_distances,
                exponential_map_w=exponential_map_w,
            )
        verts, _ = affine_mano(pose, shape, rot_6d, trans)
        for i in range(input_choirs.shape[0]):
            pl.subplot(0, i)
            V = verts[i].cpu().numpy()
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
        y_hat = model(samples["choir"])
        mano_params_gt = {
            "pose": labels["theta"],
            "beta": labels["beta"],
            "rot_6d": labels["rot"],
            "trans": labels["trans"],
        }
        mano_params_gt = {k: v[0, 0].unsqueeze(0) for k, v in mano_params_gt.items()}
        gt_pose, gt_shape, gt_rot_6d, gt_trans = tuple(mano_params_gt.values())
        gt_verts, gt_joints = affine_mano(gt_pose, gt_shape, gt_rot_6d, gt_trans)
        gt_anchors = affine_mano.get_anchors(gt_verts)
        with torch.set_grad_enabled(True):
            # Use first batch element and views as batch dimension.
            choir_pred = y_hat["choir"][0].unsqueeze(0)
            print(
                "Mean scalar: ",
                torch.mean(input_scalar).unsqueeze(0).to(input_scalar.device),
            )
            pose, shape, rot_6d, trans, anchors_pred = optimize_pose_pca_from_choir(
                choir_pred,
                bps=bps,
                scalar=torch.mean(input_scalar)
                .unsqueeze(0)
                .to(input_scalar.device),  # TODO: What should I do here?
                is_rhand=samples["is_rhand"][0],
                max_iterations=5000,
                remap_bps_distances=remap_bps_distances,
                exponential_map_w=exponential_map_w,
            )
            verts_pred, joints_pred = affine_mano(pose, shape, rot_6d, trans)
        # ============ Display the ground truth CHOIR field with the GT MANO ================
        V_gt = gt_verts[0].cpu().numpy()
        tmesh_gt = Trimesh(V_gt, F)
        gt_hand_mesh = pv.wrap(tmesh_gt)
        pl.subplot(0, input_choirs.shape[0])
        add_choir_to_plot(
            pl,
            bps / gt_scalar,
            choir_gt / gt_scalar,
            gt_ref_pts / gt_scalar,
            gt_anchors,
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
            f"Anchor error (mm): {torch.norm(anchors_pred - gt_anchors.cuda(), dim=2).mean(dim=1).mean(dim=0) * 1000:.2f}"
        )
        # === MPJPE ===
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
        # hand_mesh = pv.wrap(tmesh)

        visualize_MANO(
            tmesh, obj_ptcld=input_ref_pts[0] / input_scalar[0], gt_hand=gt_hand_mesh
        )


def visualize_model_predictions(
    model: torch.nn.Module,
    batch: Union[Tuple, List, torch.Tensor],
    step: int,
    bps: torch.Tensor,
    bps_dim: int,
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
            input_scalar,
            scalar_gt,
            rescaled_ref_pts,
            gt_rescaled_ref_pts,
            mano_params_gt,
            bps_dim=bps_dim,
        )
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
    input_scalar: float,
    gt_scalar: float,
    input_ref_pts: torch.Tensor,
    gt_ref_pts: torch.Tensor,
    gt_verts: torch.Tensor,
    gt_joints: torch.Tensor,
    gt_anchors: torch.Tensor,
    is_rhand: bool,
    use_smplx: bool,
    remap_bps_distances: bool,
    exponential_map_w: Optional[float] = None,
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
    # =============================================================
    pl = pv.Plotter(shape=(1, 2), border=False, off_screen=False)

    affine_mano = AffineMANO().cuda()
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
            choir_pred,
            bps=bps,
            scalar=input_scalar,
            is_rhand=is_rhand,
            use_smplx=use_smplx,
            max_iterations=8000,
            lr=0.01,
            remap_bps_distances=remap_bps_distances,
            exponential_map_w=exponential_map_w,
        )
    # ====== Metrics and qualitative comparison ======
    # === Anchor error ===
    print(
        f"Anchor error (mm): {torch.norm(anchors_pred - gt_anchors.cuda(), dim=2).mean(dim=1).mean(dim=0) * 1000:.2f}"
    )
    # === MPJPE ===
    if gt_joints.shape[1] == 16 and joints_pred.shape[1] == 21:
        # This was obtained with SMPL-X which doesn't include fingertips, so we'll drop them for
        # the prediction if we used MANO.
        joints_pred = torch.cat(
            [
                joints_pred[:, :4],
                joints_pred[:, 5:8],
                joints_pred[:, 9:12],
                joints_pred[:, 13:16],
                joints_pred[:, 17:20],
            ],
            dim=1,
        )

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
    pl.subplot(0, 0)
    add_choir_to_plot(
        pl,
        bps / input_scalar,
        choir_pred / input_scalar,
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
        choir_gt / gt_scalar,
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
    scalar: float,
    # dense_contacts: torch.Tensor,
    verts: torch.Tensor,
    anchors: torch.Tensor,
    # anchor_orientations: torch.Tensor,
    obj_mesh,
    obj_pointcloud,
    reference_obj_points: torch.Tensor,
    affine_mano: AffineMANO,
):
    assert len(anchors.shape) == 2
    assert len(choir.shape) == 2
    assert len(scalar.shape) == 1
    assert len(verts.shape) == 2
    if torch.is_tensor(scalar):
        scalar = scalar.item()
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
        index = i % n_anchors
        anchor = rescaled_anchors[index, :]
        # The color is proportional to the distance to the anchor. It is in hex format.
        # It is obtained from min-max normalization of the distance in the choir field,
        # without known range. The color range is 0 to 16777215.
        obj_color = int(
            (choir[i, 0] - choir[:, 0].min())
            / (choir[:, 0].max() - choir[:, 0].min())
            * 16777215
        )
        anchor_color = int(
            (choir[i, 1] - choir[:, 1].min())
            / (choir[:, 1].max() - choir[:, 1].min())
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
        # + (choir[i, 4] * anchor_orientations[i, :])
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
        index = i % n_anchors
        anchor = rescaled_anchors[index, :]
        # The color is proportional to the distance to the anchor. It is in hex format.
        # It is obtained from min-max normalization of the distance in the choir field,
        # without known range. The color range is 0 to 16777215.
        obj_color = int(
            (choir[i, 0] - choir[:, 0].min())
            / (choir[:, 0].max() - choir[:, 0].min())
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
        index = i % n_anchors
        anchor = rescaled_anchors[index, :]
        anchor_color = int(
            (choir[i, 1] - choir[:, 1].min())
            / (choir[:, 1].max() - choir[:, 1].min())
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
):
    pl = pv.Plotter(off_screen=False)
    hand_mesh = pv.wrap(pred_hand)
    pl.add_mesh(
        hand_mesh,
        opacity=0.4,
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
            opacity=0.3,
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
    pl.add_camera_orientation_widget()
    pl.add_axes_at_origin()
    pl.add_legend(loc="upper left", size=(0.1, 0.1))
    pl.show(interactive=True)
