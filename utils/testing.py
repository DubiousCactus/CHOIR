#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Eval utils.
"""

import multiprocessing
import os
from functools import partial
from typing import Any, Dict, List, Tuple

import numpy as np
import open3d as o3d
import open3d.io as o3dio
import torch
import trimesh
import trimesh.voxel.creation as voxel_create
from tqdm import tqdm

from utils import to_cuda_

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


def compute_iv_sample(
    obj_voxel_w_hand_verts: Tuple[np.ndarray, torch.Tensor],
    mesh_faces: torch.Tensor,
    pitch: float,
    radius: float,
) -> torch.Tensor:
    obj_voxel, hand_verts = obj_voxel_w_hand_verts
    hand_mesh = trimesh.Trimesh(hand_verts.numpy(), mesh_faces)
    hand_voxel = voxel_create.local_voxelize(
        hand_mesh, np.array([0, 0, 0]), pitch, radius
    )
    # If the hand voxel is empty, return 0.0:
    if np.count_nonzero(hand_voxel) == 0:
        return np.array([0.0])
    hand_voxel = hand_voxel.fill().matrix
    return np.count_nonzero((obj_voxel & hand_voxel)) * (pitch**3) * 1000000


def mp_compute_solid_intersection_volume(
    pitch: float,
    radius: float,
    batch_obj_voxels: List,
    hand_verts: torch.Tensor,
    mesh_faces: torch.Tensor,
) -> Tuple[torch.Tensor, Dict]:
    with multiprocessing.Pool(min(os.cpu_count() - 2, len(batch_obj_voxels))) as pool:
        results = tqdm(
            pool.imap(
                partial(
                    compute_iv_sample,
                    pitch=pitch,
                    radius=radius,
                    mesh_faces=mesh_faces,
                ),
                zip(
                    batch_obj_voxels,
                    [hand_verts[i].cpu() for i in range(len(batch_obj_voxels))],
                ),
            ),
            total=len(batch_obj_voxels),
            desc="Computing SIV",
        )

        # Collate the results as one tensor:
        return torch.tensor(list(results)).float().mean()


def compute_solid_intersection_volume(
    pitch: float,
    radius: float,
    batch_obj_voxels: List,
    hand_verts: torch.Tensor,
    mesh_faces: torch.Tensor,
) -> Tuple[torch.Tensor, Dict]:
    raise DeprecationWarning("Use mp_compute_solid_intersection_volume instead.")
    # TODO: This PoC works, but I need to make sure that the object mesh is always in its
    # canonical position in the test set! This might be the case for ContactPose, but probably
    # not for GRAB.
    intersection_volumes = []
    for i, obj_voxel in tqdm(
        enumerate(batch_obj_voxels), total=len(batch_obj_voxels), desc="Computing SIV"
    ):
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

    return torch.tensor(intersection_volumes).float().mean()


def process_object(
    path: str,
    center_on_obj_com: bool,
    enable_contacts_tto: bool,
    compute_iv: bool,
    pitch: float,
    radius: float,
) -> Dict[str, Dict]:
    obj_mesh = o3dio.read_triangle_mesh(path)
    if center_on_obj_com:
        obj_mesh.translate(-obj_mesh.get_center())
    N_PTS_ON_MESH = 5000
    obj_points = torch.from_numpy(
        np.asarray(obj_mesh.sample_points_uniformly(N_PTS_ON_MESH).points)
    ).float()
    obj_normals = None
    if enable_contacts_tto:
        N_NORMALS = 5000
        obj_mesh.compute_vertex_normals()
        normals_w_roots = torch.cat(
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
        random_indices = torch.randperm(normals_w_roots.shape[0])[:N_NORMALS]
        obj_normals = normals_w_roots[random_indices]
    obj_mesh = trimesh.Trimesh(vertices=obj_mesh.vertices, faces=obj_mesh.triangles)
    voxel = None
    if compute_iv:
        # TODO: Use cuda_voxelizer + libmesh's SDF-based check_mesh_contains
        voxel = (
            voxel_create.local_voxelize(obj_mesh, np.array([0, 0, 0]), pitch, radius)
            .fill()
            .matrix
        )
    obj_data = {
        "mesh": obj_mesh,
        "points": obj_points,
        "normals": obj_normals,
        "voxel": voxel,
    }
    return {path: obj_data}


def mp_process_obj_meshes(
    mesh_pths: List[str],
    obj_cache: Dict[str, Any],
    center_on_obj_com: bool,
    enable_contacts_tto: bool,
    compute_iv: bool,
    pitch: float,
    radius: float,
    keep_mesh_contact_indentity: bool = False,
):
    """
    Process a list of object meshes in parallel.
    Args:
        mesh_pths: List of paths to object meshes.
        obj_cache: Dict containing the processed object meshes.
        center_on_obj_com: Whether to center the object meshes on their center of mass.
        enable_contacts_tto: Whether to compute contact maps for the object meshes.
        compute_iv: Whether to compute the intersection volume between the object meshes and the hand meshes.
        pitch: Voxel pitch.
        radius: Voxel radius.
        keep_mesh_contact_indentity: Whether to treat all meshes as unique (they have their own contact maps) or reduce all identitical meshes to a single one.
    """
    if keep_mesh_contact_indentity:
        unique_mesh_pths = [path for path in set(mesh_pths) if path not in obj_cache]
    else:
        unique_mesh_pths = [
            path for path in set(mesh_pths) if os.path.basename(path) not in obj_cache
        ]
    n_unique_mesh_pths = len(unique_mesh_pths)
    if n_unique_mesh_pths == 0:
        return
    # print(f"[*] Processing {n_unique_mesh_pths} object meshes with {min(os.cpu_count()-2, n_unique_mesh_pths)} processes...")
    with multiprocessing.Pool(min(os.cpu_count() - 2, n_unique_mesh_pths)) as pool:
        results = tqdm(
            pool.imap(
                partial(
                    process_object,
                    center_on_obj_com=center_on_obj_com,
                    enable_contacts_tto=enable_contacts_tto,
                    compute_iv=compute_iv,
                    pitch=pitch,
                    radius=radius,
                ),
                unique_mesh_pths,
            ),
            total=n_unique_mesh_pths,
            desc="Processing object meshes",
        )

        # Collate the results as one dict:
        if keep_mesh_contact_indentity:
            obj_cache.update({k: v for d in list(results) for k, v in d.items()})
        else:
            obj_cache.update(
                {os.path.basename(k): v for d in list(results) for k, v in d.items()}
            )


def process_obj_meshes(
    mesh_pths: List[str],
    obj_cache: Dict[str, Any],
    center_on_obj_com: bool,
    enable_contacts_tto: bool,
    compute_iv: bool,
    pitch: float,
    radius: float,
):
    raise DeprecationWarning("Use mp_process_obj_meshes instead.")
    unique_mesh_pths = set(mesh_pths)
    for path in tqdm(unique_mesh_pths, desc="Processing object meshes"):
        if path in obj_cache:
            continue
        obj_mesh = o3dio.read_triangle_mesh(path)
        if center_on_obj_com:
            obj_mesh.translate(-obj_mesh.get_center())
        N_PTS_ON_MESH = 5000
        obj_points = to_cuda_(
            torch.from_numpy(
                np.asarray(obj_mesh.sample_points_uniformly(N_PTS_ON_MESH).points)
            ).float()
        )
        obj_normals = None
        if enable_contacts_tto:
            N_NORMALS = 5000
            obj_mesh.compute_vertex_normals()
            normals_w_roots = to_cuda_(
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
            random_indices = torch.randperm(normals_w_roots.shape[0])[:N_NORMALS]
            obj_normals = normals_w_roots[random_indices]
        voxel = None
        if compute_iv:
            # TODO: Use cuda_voxelizer + libmesh's SDF-based check_mesh_contains
            tmesh = trimesh.Trimesh(
                vertices=obj_mesh.vertices, faces=obj_mesh.triangles
            )
            voxel = (
                voxel_create.local_voxelize(tmesh, np.array([0, 0, 0]), pitch, radius)
                .fill()
                .matrix
            )
        obj_data = {
            "mesh": obj_mesh,
            "points": obj_points,
            "normals": obj_normals,
            "voxel": voxel,
        }
        obj_cache[path] = obj_data


def make_batch_of_obj_data(
    obj_data_cache: Dict[str, Any],
    mesh_pths: List[str],
    keep_mesh_contact_indentity: bool = False,
) -> Dict[str, torch.Tensor]:
    batch_obj_data = {}
    if not keep_mesh_contact_indentity:
        mesh_pths = [os.path.basename(path) for path in mesh_pths]
    for path in mesh_pths:
        for k, v in obj_data_cache[path].items():
            if k not in batch_obj_data:
                batch_obj_data[k] = []
            batch_obj_data[k].append(v)
    for k, v in batch_obj_data.items():
        batch_obj_data[k] = (
            to_cuda_(torch.stack(v, dim=0)) if type(v[0]) == torch.Tensor else v
        )
    return batch_obj_data
