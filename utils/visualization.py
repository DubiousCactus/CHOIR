#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Visualization utilities.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import open3d
import pyvista as pv
import torch
from bps_torch.bps import bps_torch
from bps_torch.tools import denormalize
from trimesh import Trimesh

import conf.project as project_conf
from model.affine_mano import AffineMANO
from utils import to_cuda
from utils.training import optimize_pose_pca_from_choir


def visualize_model_predictions(
    model: torch.nn.Module, batch: Union[Tuple, List, torch.Tensor], step: int, **kwargs
) -> None:
    x, y = batch  # type: ignore
    noisy_choir, pcl_mean, pcl_scalar, bps_dim = x
    (
        choir_gt,
        anchor_deltas,
        joints_gt,
        anchors_gt,
        pose_gt,
        beta_gt,
        rot_gt,
        trans_gt,
    ) = y
    if not project_conf.HEADLESS:
        noisy_choir, pcl_mean, pcl_scalar = (
            noisy_choir,
            pcl_mean,
            pcl_scalar,
        )
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
            pcl_mean,
            pcl_scalar,
            mano_params_gt,
            bps_dim=bps_dim.squeeze()[0].long().item(),
            anchor_assignment=kwargs["anchor_assignment"],
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
    pcl_mean: torch.Tensor,
    pcl_scalar: torch.Tensor,
    mano_params_gt: Dict[str, torch.Tensor],
    bps_dim: int,
    anchor_assignment: str,
):
    def add_choir_to_plot(plot, choir, hand_mesh, anchors):
        reference_obj_points = bps.decode(x_deltas=choir[:, :, 1:4])
        reference_obj_points = denormalize(reference_obj_points, pcl_mean, pcl_scalar)
        plot.add_points(
            reference_obj_points[0].cpu().numpy(),
            color="blue",
            name="target_points",
            opacity=0.9,
        )
        if anchor_assignment == "closest_and_farthest":
            min_dist = min(choir[0, :, -33].min(), choir[0, :, 4].min())
            max_dist = max(choir[0, :, -33].max(), choir[0, :, 4].max())
        elif anchor_assignment in ["closest", "random", "batched_fixed"]:
            min_dist, max_dist = choir[0, :, 4].min(), choir[0, :, 4].max()
        else:
            raise NotImplementedError
        max_dist = max_dist + 1e-6
        for i in range(0, reference_obj_points.shape[1], 1):
            # The color is proportional to the distance to the anchor. It is in hex format.
            # It is obtained from min-max normalization of the distance in the choir field,
            # without known range. The color range is 0 to 16777215.
            if anchor_assignment in [
                "closest",
                "closest_and_farthest",
                "random",
                "batched_fixed",
            ]:
                if anchor_assignment == "batched_fixed":
                    assert (
                        bps_dim % 32 == 0
                    ), "bps_dim must be a multiple of 32 for batched_fixed anchor assignment."
                    index = i % 32
                else:
                    choir_one_hot = choir[0, i, 5:37]
                    index = torch.argmax(choir_one_hot, dim=-1)
                anchor = anchors[0, index, :]
                color = int(
                    (choir[0, i, 4] - min_dist) / (max_dist - min_dist) * 16777215
                )
                plot.add_lines(
                    np.array(
                        [
                            reference_obj_points[0, i, :].cpu().numpy(),
                            anchor.cpu().numpy(),
                        ]
                    ),
                    width=1,
                    color="#" + hex(color)[2:].zfill(6),
                    name=f"closest_ray{i}",
                )
                if anchor_assignment == "closest_and_farthest":
                    choir_one_hot = choir[0, i, -32:]
                    index = torch.argmax(choir_one_hot, dim=-1)
                    farthest_anchor = anchors[0, index, :]
                    color = int(
                        (choir[0, i, -33] - min_dist) / (max_dist - min_dist) * 16777215
                    )
                    plot.add_lines(
                        np.array(
                            [
                                reference_obj_points[0, i, :].cpu().numpy(),
                                farthest_anchor.cpu().numpy(),
                            ]
                        ),
                        width=1,
                        color="#" + hex(color)[2:].zfill(6),
                        name=f"farthest_ray{i}",
                    )
            else:
                raise NotImplementedError
        plot.add_mesh(
            hand_mesh,
            opacity=0.4,
            name="hand_mesh",
            smooth_shading=True,
        )
        for i in range(anchors.shape[1]):
            plot.add_mesh(
                pv.Cube(
                    center=anchors[0, i].cpu().numpy(),
                    x_length=3e-3,
                    y_length=3e-3,
                    z_length=3e-3,
                ),
                color="yellow",
                name=f"anchor{i}",
            )

    # ============ Get the first element of the batch ============
    choir_pred = choir_pred[0].unsqueeze(0)
    choir_gt = choir_gt[0].unsqueeze(0)
    pcl_mean = pcl_mean[0].unsqueeze(0)
    pcl_scalar = pcl_scalar[0].unsqueeze(0)
    mano_params_gt = {k: v[0].unsqueeze(0) for k, v in mano_params_gt.items()}
    # =============================================================

    pl = pv.Plotter(shape=(1, 2), border=False, off_screen=False)
    pl.subplot(0, 0)

    bps = bps_torch(
        bps_type="random_uniform",
        n_bps_points=bps_dim,
        radius=1.0,
        n_dims=3,
        custom_basis=None,
    )
    affine_mano = AffineMANO().cuda()
    faces = affine_mano.faces
    F = faces.cpu().numpy()
    # ============ Optimize MANO on the predicted CHOIR field to visualize it ============
    with torch.set_grad_enabled(True):
        pose, shape, rot_6d, trans, anchors_pred = optimize_pose_pca_from_choir(
            choir_pred,
            anchor_assignment=anchor_assignment,
            hand_contacts=None,
            bps_dim=bps_dim,
            x_mean=pcl_mean,
            x_scalar=pcl_scalar,
            objective="anchors",
            max_iterations=1000,
        )
    # ====== Metrics and qualitative comparison ======
    gt_pose, gt_shape, gt_rot_6d, gt_trans = tuple(mano_params_gt.values())
    gt_verts, gt_joints = affine_mano(gt_pose, gt_shape, gt_rot_6d, gt_trans)
    gt_anchors = affine_mano.get_anchors(gt_verts)
    verts_pred, joints_pred = affine_mano(pose, shape, rot_6d, trans)
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
    hand_mesh = pv.wrap(tmesh)
    add_choir_to_plot(pl, choir_pred, hand_mesh, anchors_pred)
    # ===================================================================================
    # ============ Display the ground truth CHOIR field with the GT MANO ================
    pl.subplot(0, 1)
    V = gt_verts[0].cpu().numpy()
    tmesh = Trimesh(V, F)
    gt_hand_mesh = pv.wrap(tmesh)
    add_choir_to_plot(pl, choir_gt, gt_hand_mesh, gt_anchors)
    pl.link_views()
    pl.set_background("white")  # type: ignore
    pl.add_camera_orientation_widget()
    pl.show(interactive=True)


def visualize_CHOIR(
    choir: torch.Tensor,
    dense_contacts: torch.Tensor,
    verts: torch.Tensor,
    anchors: torch.Tensor,
    anchor_orientations: torch.Tensor,
    obj_mesh,
    obj_pointcloud,
    reference_obj_points: torch.Tensor,
    affine_mano: AffineMANO,
    anchor_assignment: str,
):
    n_anchors = anchors.shape[0]
    faces = affine_mano.faces
    V = verts[0].cpu().numpy()
    F = faces.cpu().numpy()
    tmesh = Trimesh(V, F)
    hand_mesh = pv.wrap(tmesh)

    pl = pv.Plotter(shape=(1, 2), border=False, off_screen=False)
    pl.subplot(0, 0)
    pl.add_mesh(
        hand_mesh,
        opacity=0.4,
        name="hand_mesh",
        smooth_shading=True,
        scalars=dense_contacts.cpu().numpy(),
        cmap="jet",
    )
    obj_tmesh = Trimesh(obj_mesh.vertices, obj_mesh.triangles)
    obj_mesh_pv = pv.wrap(obj_tmesh)
    pl.add_mesh(
        obj_mesh_pv, opacity=0.3, name="obj_mesh", smooth_shading=True, color="red"
    )
    for i in range(n_anchors):
        pl.add_mesh(
            pv.Cube(
                center=anchors[i].cpu().numpy(),
                x_length=3e-3,
                y_length=3e-3,
                z_length=3e-3,
            ),
            color="yellow",
            name=f"anchor{i}",
        )

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
        obj_pointcloud.cpu().numpy(), color="red", name="base_ptcld", opacity=0.2
    )
    bps_dim = reference_obj_points.shape[0]
    for i in range(0, reference_obj_points.shape[0], 1):
        if anchor_assignment in ["closest", "random"]:
            index = torch.argmax(choir[i, -32:])
        elif anchor_assignment == "batched_fixed":
            index = i % n_anchors
        else:
            raise NotImplementedError(
                f"Anchor assignment {anchor_assignment} not implemented."
            )
        anchor = anchors[index, :]
        # The color is proportional to the distance to the anchor. It is in hex format.
        # It is obtained from min-max normalization of the distance in the choir field,
        # without known range. The color range is 0 to 16777215.
        color = int(
            (choir[i, 4] - choir[:, 4].min())
            / (choir[:, 4].max() - choir[:, 4].min())
            * 16777215
        )
        # This is to check that the delta vectors are correct: (should be the same visual
        # result as above)
        assert np.allclose(
            np.array([reference_obj_points[i, :].cpu().numpy(), anchor.cpu().numpy()]),
            np.array(
                [
                    reference_obj_points[i, :].cpu().numpy(),
                    (
                        reference_obj_points[i, :].cpu()
                        + (choir[i, 4] * anchor_orientations[i, :])
                    ).numpy(),
                ]
            ),
        )
        pl.add_lines(
            np.array([reference_obj_points[i, :].cpu().numpy(), anchor.cpu().numpy()]),
            width=1,
            color="#" + hex(color)[2:].zfill(6),
            name=f"ray{i}",
        )

    for i in range(n_anchors):
        pl.add_mesh(
            pv.Cube(
                center=anchors[i].cpu().numpy(),
                x_length=3e-3,
                y_length=3e-3,
                z_length=3e-3,
            ),
            color="yellow",
            name=f"anchor{i}",
        )
    pl.link_views()
    pl.set_background("white")  # type: ignore
    pl.add_camera_orientation_widget()
    pl.show(interactive=True)


def visualize_MANO(
    verts: torch.Tensor,
    faces: torch.Tensor,
    obj_mesh: Optional[open3d.geometry.TriangleMesh] = None,
    gt_hand: Optional[Any] = None,
):
    pl = pv.Plotter(off_screen=False)
    V = verts[0].detach().cpu().numpy()
    F = faces.cpu().numpy()  # type: ignore
    tmesh = Trimesh(V, F)
    hand_mesh = pv.wrap(tmesh)
    pl.add_mesh(hand_mesh, opacity=0.4, name="hand_mesh", smooth_shading=True)
    if gt_hand is not None:
        pl.add_mesh(
            gt_hand, opacity=0.2, name="gt_hand", smooth_shading=True, color="blue"
        )
    if obj_mesh is not None:
        obj_tmesh = Trimesh(obj_mesh.vertices, obj_mesh.triangles)
        obj_mesh_pv = pv.wrap(obj_tmesh)
        pl.add_mesh(
            obj_mesh_pv, opacity=0.3, name="obj_mesh", smooth_shading=True, color="red"
        )
    pl.set_background("white")  # type: ignore
    pl.add_camera_orientation_widget()
    pl.show(interactive=True)
