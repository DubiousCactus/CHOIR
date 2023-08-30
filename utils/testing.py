#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Eval utils.
"""

from typing import Dict, List, Tuple

import numpy as np
import open3d as o3d
import open3d.io as o3dio
import torch
import trimesh
import trimesh.voxel.creation as voxel_create
from tqdm import tqdm

""" Taken from https://github.com/shreyashampali/ho3d/blob/master/eval.py#L52 """


def verts2pcd(verts, color=None):
    pcd = o3d.PointCloud()
    pcd.points = o3d.Vector3dVector(verts)
    if color is not None:
        if color == "r":
            pcd.paint_uniform_color([1, 0.0, 0])
        if color == "g":
            pcd.paint_uniform_color([0, 1.0, 0])
        if color == "b":
            pcd.paint_uniform_color([0, 0, 1.0])
    return pcd


def calculate_fscore(gt, pr, th=0.01):
    gt = verts2pcd(gt)
    pr = verts2pcd(pr)
    d1 = o3d.compute_point_cloud_to_point_cloud_distance(
        gt, pr
    )  # closest dist for each gt point
    d2 = o3d.compute_point_cloud_to_point_cloud_distance(
        pr, gt
    )  # closest dist for each pred point
    if len(d1) and len(d2):
        recall = float(sum(d < th for d in d2)) / float(
            len(d2)
        )  # how many of our predicted points lie close to a gt point?
        precision = float(sum(d < th for d in d1)) / float(
            len(d1)
        )  # how many of gt points are matched?

        if recall + precision > 0:
            fscore = 2 * recall * precision / (recall + precision)
        else:
            fscore = 0
    else:
        fscore = 0
        precision = 0
        recall = 0
    return fscore, precision, recall


""" ============================================================================= """


def compute_mpjpe(
    gt_joints: torch.Tensor, pred_joints: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    pjpe = torch.linalg.vector_norm(
        gt_joints - pred_joints, ord=2, dim=-1
    )  # Per-joint position error (B, 21)
    mpjpe = torch.mean(pjpe, dim=-1)  # Mean per-joint position error (B, 1)
    mpjpe = torch.mean(
        mpjpe, dim=0
    )  # Mean per-joint position error avgd across batch (1)
    root_aligned_pjpe = torch.linalg.vector_norm(
        (gt_joints - gt_joints[:, 0:1, :]) - (pred_joints - pred_joints[:, 0:1, :]),
        ord=2,
        dim=-1,
    )  # Per-joint position error (B, 21)
    root_aligned_mpjpe = torch.mean(
        root_aligned_pjpe, dim=-1
    )  # Mean per-joint position error (B, 1)
    root_aligned_mpjpe = torch.mean(
        root_aligned_mpjpe, dim=0
    )  # Mean per-joint position error avgd across batch (1)
    return mpjpe, root_aligned_mpjpe


def compute_solid_intersection_volume_other():
    pass
    # Compute the intersection volume between the predicted hand meshes and the object meshes.
    # TODO: Implement it with another method cause it crashes (see
    # https://github.com/isl-org/Open3D/issues/5911)
    # intersection_volumes = []
    # mano_faces = self._affine_mano.faces.cpu().numpy()
    # for i, path in enumerate(mesh_pths):
    # obj_mesh = o3dtg.TriangleMesh.from_legacy(o3dio.read_triangle_mesh(path))
    # hand_mesh = o3dtg.TriangleMesh.from_legacy(
    # o3dg.TriangleMesh(
    # o3du.Vector3dVector(verts_pred[i].cpu().numpy()),
    # o3du.Vector3iVector(mano_faces),
    # )
    # )
    # intersection = obj_mesh.boolean_intersection(hand_mesh)
    # intersection_volumes.append(intersection.to_legacy().get_volume())
    # intersection_volume = torch.tensor(intersection_volumes).mean()
    # intersection_volume *= (
    # self._data_loader.dataset.base_unit / 10
    # )  # m^3 -> mm^3 -> cm^3
    # We'll do the same with trimesh.boolean.intersection():
    # intersection_volumes = []
    # mano_faces = self._affine_mano.closed_faces.cpu().numpy()
    # print("[*] Computing intersection volumes...")
    # for i, path in tqdm(enumerate(mesh_pths), total=len(mesh_pths)):
    # obj_mesh = o3dio.read_triangle_mesh(path)
    # if self._data_loader.dataset.center_on_object_com:
    # obj_mesh.translate(-obj_mesh.get_center())
    # obj_mesh = trimesh.Trimesh(
    # vertices=obj_mesh.vertices, faces=obj_mesh.triangles
    # )
    # hand_mesh = trimesh.Trimesh(verts_pred[i].cpu().numpy(), mano_faces)
    # # TODO: Recenter the obj_mesh and potentially rescale it. Visualize to check.
    # # visualize_MANO(hand_mesh, obj_mesh=obj_mesh)
    # intersection = obj_mesh.intersection(hand_mesh)
    # if intersection.volume > 0:
    # print(self._data_loader.dataset.base_unit)
    # print(f"{intersection.volume * (self._data_loader.dataset.base_unit/10):.2f} cm^3")
    # print(f"Is water tight? {intersection.is_watertight}")
    # visualize_MANO(hand_mesh, obj_mesh=obj_mesh)
    # visualize_MANO(hand_mesh, obj_mesh=intersection)
    # intersection_volumes.append(intersection.volume)


def compute_solid_intersection_volume(
    pitch: float,
    radius: float,
    mesh_pths: List[str],
    hand_verts: torch.Tensor,
    mesh_faces: torch.Tensor,
    center_on_object_com: bool,
    return_meshes: bool = False,
) -> Tuple[torch.Tensor, Dict]:
    # TODO: This PoC works, but I need to make sure that the object mesh is always in its
    # canonical position in the test set! This might be the case for ContactPose, but probably
    # not for GRAB.
    # TODO: Multithread this?
    # TODO: Do the same with PyVista or Open3D? Whatever is fastest because this is slooooow.
    intersection_volumes = []
    obj_voxels = {}
    obj_meshes = {}
    for i, path in tqdm(enumerate(mesh_pths), total=len(mesh_pths)):
        if path not in obj_voxels:
            obj_mesh = o3dio.read_triangle_mesh(path)
            if center_on_object_com:
                obj_mesh.translate(-obj_mesh.get_center())

            if return_meshes:
                obj_meshes[path] = obj_mesh

            obj_mesh = trimesh.Trimesh(
                vertices=obj_mesh.vertices, faces=obj_mesh.triangles
            )
            obj_voxel = (
                voxel_create.local_voxelize(
                    obj_mesh, np.array([0, 0, 0]), pitch, radius
                )
                .fill()
                .matrix
            )
            obj_voxels[path] = obj_voxel
        else:
            obj_voxel = obj_voxels[path]
        hand_mesh = trimesh.Trimesh(hand_verts[i].cpu().numpy(), mesh_faces)
        hand_voxel = voxel_create.local_voxelize(
            hand_mesh, np.array([0, 0, 0]), pitch, radius
        )
        # If the hand voxel is empty, return 0.0:
        if np.count_nonzero(hand_voxel) == 0:
            intersection_volumes.append(0.0)
            continue
        hand_voxel = hand_voxel.fill().matrix
        # both_voxels = trimesh.voxel.VoxelGrid(
        # trimesh.voxel.encoding.DenseEncoding(
        # obj_voxel | hand_voxel
        # ),
        # )
        # both_voxels.show()
        # obj_volume = (
        # np.count_nonzero(obj_voxel) * (pitch**3) * 1000000
        # )  # m^3 -> cm^3
        # hand_volume = (
        # np.count_nonzero(hand_voxel) * (pitch**3) * 1000000
        # )  # m^3 -> cm^3
        # typical_hand_volume = (
        # 379.7  # cm^3 https://doi.org/10.1177/154193128603000417
        # )
        # Make sure we're within 35% of the typical hand volume:
        # assert (
        # hand_volume > typical_hand_volume * 0.65
        # and hand_volume < typical_hand_volume * 1.35
        # ), f"Hand volume is {hand_volume:.2f} cm^3, which is not within 30% of the typical"
        # if (
        # hand_volume > typical_hand_volume * 0.65
        # and hand_volume < typical_hand_volume * 1.35
        # ):
        # print(
        # f"Hand volume is {hand_volume:.2f} cm^3, which is not within 30% of the typical"
        # )
        intersection_volume = (
            np.count_nonzero((obj_voxel & hand_voxel)) * (pitch**3) * 1000000
        )
        # print(
        # f"Volume of hand: {hand_volume:.2f} cm^3, volume of obj: {obj_volume:.2f} cm^3, intersection volume: {intersection_volume:.2f} cm^3"
        # )
        intersection_volumes.append(intersection_volume)

    return (
        torch.tensor(intersection_volumes).float().mean(),
        obj_meshes if return_meshes else None,
    )
