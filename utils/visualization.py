#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Visualization utilities.
"""

from typing import Any, Optional

import numpy as np
import open3d
import pyvista as pv
import torch
from trimesh import Trimesh

from model.affine_mano import AffineMANO


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
