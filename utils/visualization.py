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
    model: torch.nn.Module, batch: Union[Tuple, List, torch.Tensor], step: int
) -> None:
    x, y = batch  # type: ignore
    if not project_conf.HEADLESS:
        noisy_choir, pcl_mean, pcl_scalar = (
            x["noisy_choir"],
            x["pcl_mean"],
            x["pcl_scalar"],
        )
        choir_pred = model(noisy_choir)
        gt_choir, mano_params_gt = y["choir"], y["mano_params"]
        visualize_CHOIR_prediction(
            choir_pred,
            gt_choir,
            pcl_mean,
            pcl_scalar,
            mano_params_gt,
            bps_dim=x["bps_dim"].squeeze()[0].long().item(),
        )
    if project_conf.USE_WANDB:
        # TODO: Log a few predictions and the ground truth to wandb.
        # wandb.log({"pointcloud": wandb.Object3D(ptcld)}, step=step)
        raise NotImplementedError("Visualization is not implemented for wandb.")


@to_cuda
def visualize_CHOIR_prediction(
    choir_pred: torch.Tensor,
    choir_gt: torch.Tensor,
    pcl_mean: torch.Tensor,
    pcl_scalar: torch.Tensor,
    mano_params_gt: Dict[str, torch.Tensor],
    bps_dim: int,
):
    def add_choir_to_plot(plot, choir, hand_mesh):
        reference_obj_points = bps.decode(x_deltas=choir[:, :, 1:4])
        reference_obj_points = denormalize(reference_obj_points, pcl_mean, pcl_scalar)
        plot.add_points(
            reference_obj_points.cpu().numpy(),
            color="blue",
            name="target_points",
            opacity=0.9,
        )
        for i in range(reference_obj_points.shape[1]):
            anchor = anchors[0, int(choir[0, i, -1]), :]
            # The color is proportional to the distance to the anchor. It is in hex format.
            # It is obtained from min-max normalization of the distance in the choir field,
            # without known range. The color range is 0 to 16777215.
            color = int(
                (choir[0, i, 0] - choir[0, :, 0].min())
                / (choir[0, :, 0].max() - choir[0, :, 0].min())
                * 16777215
            )
            plot.add_lines(
                np.array(
                    [reference_obj_points[0, i, :].cpu().numpy(), anchor.cpu().numpy()]
                ),
                width=1,
                color="#" + hex(color)[2:].zfill(6),
                name=f"ray{i}",
            )
        plot.add_mesh(
            hand_mesh,
            opacity=0.4,
            name="hand_mesh",
            smooth_shading=True,
        )

    # ============ Get the first element of the batch ============
    choir_pred = choir_pred[0].unsqueeze(0)
    choir_gt = choir_gt[0].unsqueeze(0)
    pcl_mean = pcl_mean[0].unsqueeze(0)
    pcl_scalar = pcl_scalar[0].unsqueeze(0)
    print(mano_params_gt)
    mano_params_gt = {k: v[0].unsqueeze(0) for k, v in mano_params_gt.items()}
    print(mano_params_gt)
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
        pose, shape, rot_6d, trans, anchors = optimize_pose_pca_from_choir(
            choir_pred,
            hand_contacts=None,
            bps_dim=bps_dim,
            x_mean=pcl_mean,
            x_scalar=pcl_scalar,
            objective="anchors",
        )
    verts, _ = affine_mano(pose, shape, rot_6d, trans)
    V = verts[0].cpu().numpy()
    tmesh = Trimesh(V, F)
    hand_mesh = pv.wrap(tmesh)
    add_choir_to_plot(pl, choir_pred, hand_mesh)
    # ===================================================================================
    # ============ Display the ground truth CHOIR field with the GT MANO ================
    pl.subplot(0, 1)
    pose, shape, rot_6d, trans = tuple(mano_params_gt.values())
    print(pose.shape, shape.shape)
    verts, _ = affine_mano(pose, shape, rot_6d, trans)
    V = verts[0].cpu().numpy()
    tmesh = Trimesh(V, F)
    hand_mesh = pv.wrap(tmesh)
    add_choir_to_plot(pl, choir_gt, hand_mesh)
    pl.link_views()
    pl.set_background("white")  # type: ignore
    pl.add_camera_orientation_widget()
    pl.show(interactive=True)


def visualize_CHOIR(
    choir: torch.Tensor,
    dense_contacts: torch.Tensor,
    verts: torch.Tensor,
    anchors: torch.Tensor,
    obj_mesh,
    obj_pointcloud,
    reference_obj_points: torch.Tensor,
    affine_mano: AffineMANO,
):
    n_anchors = anchors.shape[1]
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
                center=anchors[0, i].cpu().numpy(),
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
    for i in range(0, reference_obj_points.shape[0], 1):
        anchor = anchors[0, int(choir[0, i, -1]), :]
        # The color is proportional to the distance to the anchor. It is in hex format.
        # It is obtained from min-max normalization of the distance in the choir field,
        # without known range. The color range is 0 to 16777215.
        color = int(
            (choir[0, i, 0] - choir[0, :, 0].min())
            / (choir[0, :, 0].max() - choir[0, :, 0].min())
            * 16777215
        )
        pl.add_lines(
            np.array([reference_obj_points[i, :].cpu().numpy(), anchor.cpu().numpy()]),
            width=1,
            color="#" + hex(color)[2:].zfill(6),
            name=f"ray{i}",
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
