#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Visualization utilities.
"""

import copy
from typing import Any, List, Optional, Tuple, Union

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
    samples, labels, mesh_paths = batch
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
        mesh_paths = mesh_paths[-1]  # Now we have a list of B entries.
        N_PTS_ON_MESH = 3000
        obj_mesh = open3d.io.read_triangle_mesh(mesh_paths[0])
        obj_normals = to_cuda_(
            torch.cat(
                (
                    torch.from_numpy(np.asarray(obj_mesh.vertices)).type(
                        dtype=torch.float32
                    ),
                    torch.from_numpy(np.asarray(obj_mesh.vertex_normals)).type(
                        dtype=torch.float32
                    ),
                ),
                dim=-1,
            )
        )
        obj_points = to_cuda_(
            torch.from_numpy(
                np.asarray(obj_mesh.sample_points_uniformly(N_PTS_ON_MESH).points)
            ).float()
        )
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
        elif kwargs.get("method", "aggved") == "coddpm":
            conditional = kwargs["conditional"]
            n = 1
            choir_pred, contacts_pred = model.generate(
                n,
                y=samples["choir"][0].unsqueeze(0) if conditional else None,
            )
            choir_pred, contacts_pred = choir_pred.squeeze(1), contacts_pred.squeeze(1)
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
                    contact_gaussians=contacts_pred,
                    obj_pts=obj_points,
                    obj_normals=obj_normals,
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




@to_cuda


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
    pl.add_arrows(
        cent=np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
        direction=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        mag=0.1,
    )
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
            opacity=0.8,
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
            # opacity=opacity,
            name="obj_mesh",
            label="Object mesh",
            smooth_shading=True,
            color="red",
        )
    elif obj_ptcld is not None:
        pl.add_points(
            obj_ptcld.detach().cpu().numpy()
            if torch.is_tensor(obj_ptcld)
            else obj_ptcld,
            color="red",
            name="obj_ptcld",
            label="Object pointcloud",
            opacity=0.2,
        )
    # pl.add_title("Fitted MANO vs ground-truth MANO", font_size=30)
    pl.set_background("white")  # type: ignore
    # pl.add_camera_orientation_widget()
    # pl.add_legend(loc="upper left", size=(0.1, 0.1))
    if cam_pose is not None:
        pl.camera.model_transform_matrix = cam_pose[1]
        pl.camera.position = cam_pose[0]
    if save_as is not None:
        # pl.show()
        # pl.screenshot(save_as)
        # pl.save_graphic(save_as)
        pl.export_html(save_as)
    else:
        # pl.add_axes_at_origin()
        # We need smaller axes, so let's do it manually:
        pl.add_arrows(
            cent=np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
            direction=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            mag=0.1,
        )
        pl.show(interactive=True)
    if return_cam_pose:
        return (pl.camera.position, pl.camera.model_transform_matrix)


def visualize_3D_gaussians_on_hand_mesh(
    hand_mesh: Trimesh,
    obj_mesh: Trimesh,
    gaussian_params: torch.Tensor,
    base_unit: float,
    anchors: torch.Tensor,
    debug_anchor: Optional[int] = None,
    anchor_contacts: Optional[torch.Tensor] = None,
):
    geometries = []
    for i in range(gaussian_params.shape[0]):
        if debug_anchor is not None and i != debug_anchor:
            continue
        # print(f"Visualizing Gaussian {i + 1} / {gaussian_params.shape[0]}")
        if torch.all(gaussian_params[i] == 0):
            continue
        mean = gaussian_params[i, :3] + anchors[i]
        covariance = gaussian_params[i, 3:].reshape(3, 3)
        # print(covariance)
        # Sample points from the Gaussian
        num_points = 10000
        points = np.random.multivariate_normal(mean, covariance, num_points)

        # Create a point cloud from the sampled points
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(points)
        # Partition RGB colourspace in 32 and use a different colour for each anchor:
        pcd.paint_uniform_color(
            np.array(
                [
                    (int((i * 16) % 255) / 255.0) if i < 32 / 3 else 1,
                    (int((i * 16) % 255) / 255.0)
                    if (i >= 32 / 3 and i < 64 / 3)
                    else 5,
                    (int((i * 16) % 255) / 255.0) if i >= 64 / 3 else 0,
                ]
            )
            if debug_anchor is None
            else np.array([0, 0, 1])
        )
        geometries.append(pcd)

    if type(hand_mesh) == Trimesh:
        o3d_hand_mesh = open3d.geometry.TriangleMesh()
        o3d_hand_mesh.vertices = open3d.utility.Vector3dVector(hand_mesh.vertices)
        o3d_hand_mesh.triangles = open3d.utility.Vector3iVector(hand_mesh.faces)
        geometries.append(o3d_hand_mesh)
    else:
        geometries.append(hand_mesh)
    # geometries.append(obj_mesh)
    if debug_anchor is not None:
        assert anchors is not None
        # Draw a green cube around the debug anchor
        debug_color = [0.04, 0.66, 0.43]
        box = open3d.geometry.TriangleMesh.create_box(
            5 / base_unit, 5 / base_unit, 5 / base_unit
        ).translate(anchors[debug_anchor] - np.array([2.5, 2.5, 2.5]) / base_unit)
        box.paint_uniform_color(debug_color)
        geometries.append(box)
        print(
            f"Anchor {debug_anchor} has vertices: {anchor_contacts[debug_anchor][0].shape}, "
            + f"contact-weighted vertices={anchor_contacts[debug_anchor][1].shape}"
            + f" and total contact points={anchor_contacts[debug_anchor][2].sum()}"
        )
        pcd = open3d.geometry.PointCloud()
        pts = anchor_contacts[debug_anchor][0].cpu().numpy()
        # Remove the pts that have a contact value of 0:
        # pts = pts[contacts > 0]
        pcd.points = open3d.utility.Vector3dVector(pts)
        pcd.paint_uniform_color(debug_color)
        geometries.append(pcd)
    # Visualize the point cloud
    open3d.visualization.draw_geometries(geometries)


def visualize_hand_contacts_from_3D_gaussians(
    hand_mesh: Trimesh,
    gaussian_params: torch.Tensor,
    anchors: torch.Tensor,
    gt_contacts: Optional[torch.Tensor] = None,
    return_trimesh: bool = False,
):
    geometries = []
    vertex_contacts = torch.zeros(np.asarray(hand_mesh.vertices).shape[0])

    hand_verts = torch.from_numpy(np.asarray(hand_mesh.vertices)).float()
    anchors = anchors.float()
    anchor_distances = torch.cdist(hand_verts, anchors)  # (V, A)
    anchor_distances, anchor_indices = torch.topk(
        anchor_distances, 1, dim=-1, largest=False, sorted=False
    )
    scale_tril_norm_activation_threshold = 1e-3
    for i in range(gaussian_params.shape[0]):
        if torch.allclose(
            gaussian_params[i], torch.zeros_like(gaussian_params[i]), atol=1e-6
        ):
            continue
        if (
            torch.norm(gaussian_params[i, 3:])
            < scale_tril_norm_activation_threshold**2
        ):
            continue
        mean = gaussian_params[i, :3] + anchors[i]
        covariance = (
            gaussian_params[i, 3:].reshape(3, 3) + 1e-9 * torch.eye(3)[None, ...]
        )
        gaussian = torch.distributions.MultivariateNormal(mean, covariance)
        this_anchor_indices = (anchor_indices == i).squeeze(-1)
        anchor_verts = hand_verts[this_anchor_indices]
        # vertex_contacts[this_anchor_indices] = torch.clamp_max(torch.exp(gaussian.log_prob(anchor_verts)), 1.0)
        contact_vals = torch.exp(gaussian.log_prob(anchor_verts))
        if contact_vals.max() == 0:
            continue
        print(f"Max contact values for Gaussian {i}/32: {contact_vals.max()}")
        vertex_contacts[this_anchor_indices] = contact_vals / contact_vals.max()

    colours = np.zeros_like(hand_mesh.vertices)
    colours[:, 0] = vertex_contacts  # / vertex_contacts.max()
    colours[:, 1] = 0.58 - colours[:, 0]
    colours[:, 2] = 0.66 - colours[:, 0]
    if type(hand_mesh) == Trimesh:
        o3d_hand_mesh = open3d.geometry.TriangleMesh()
        o3d_hand_mesh.vertices = open3d.utility.Vector3dVector(hand_mesh.vertices)
        o3d_hand_mesh.triangles = open3d.utility.Vector3iVector(hand_mesh.faces)
        o3d_hand_mesh.vertex_colors = open3d.utility.Vector3dVector(colours)
        geometries.append(o3d_hand_mesh)
    else:
        hand_mesh.vertex_colors = open3d.utility.Vector3dVector(colours)
        geometries.append(hand_mesh)

    if gt_contacts is not None:
        gt_hand_mesh = copy.deepcopy(hand_mesh)
        gt_colours = np.zeros_like(hand_mesh.vertices)
        gt_colours[:, 0] = gt_contacts / gt_contacts.max()
        gt_colours[:, 1] = 0.04 - colours[:, 0]
        gt_colours[:, 2] = 0.77 - colours[:, 0]
        gt_hand_mesh.vertex_colors = open3d.utility.Vector3dVector(gt_colours)
        gt_hand_mesh.translate([0.1, 0, 0])
        geometries.append(gt_hand_mesh)
    if return_trimesh:
        return Trimesh(hand_mesh.vertices, hand_mesh.faces, vertex_colors=colours)
    open3d.visualization.draw_geometries(geometries)
