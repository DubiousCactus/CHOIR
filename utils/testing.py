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
from typing import Any, Dict, List, Optional, Tuple

import pickle
import numpy as np
import open3d as o3d
import open3d.io as o3dio
import open3d.geometry as o3dg
import open3d.utility as o3du
import torch
import trimesh
import trimesh.voxel.creation as voxel_create
from manotorch.upsamplelayer import UpSampleLayer
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
        return 0.0
    hand_voxel = hand_voxel.fill().matrix
    return np.count_nonzero((obj_voxel & hand_voxel)) * (pitch**3) * 1000000


def mp_compute_solid_intersection_volume(
    batch_obj_voxels: List,
    hand_verts: torch.Tensor,
    mesh_faces: torch.Tensor,
    pitch: float,
    radius: float,
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


def process_object(
    path: str,
    center_on_obj_com: bool,
    enable_contacts_tto: bool,
    compute_iv: bool,
    pitch: float,
    radius: float,
    dataset: str,
    n_samples: int = 5000,
    n_normals: int = 5000,
) -> Dict[str, Dict]:
    if dataset == "oakink":
        with open(path, "rb") as f:
            obj_dict = pickle.load(f)
        obj_mesh = o3dg.TriangleMesh()
        obj_mesh.vertices = o3du.Vector3dVector(obj_dict["verts"])
        obj_mesh.triangles = o3du.Vector3iVector(obj_dict["faces"])
    else:
        obj_mesh = o3dio.read_triangle_mesh(path)
    if center_on_obj_com:
        obj_mesh.translate(-obj_mesh.get_center())
    obj_points = torch.from_numpy(
        np.asarray(obj_mesh.sample_points_uniformly(n_samples).points)
    ).float()
    obj_normals = None
    if enable_contacts_tto:
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
        random_indices = torch.randperm(normals_w_roots.shape[0])[:n_normals]
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
    n_samples: int,
    n_normals: int,
    keep_mesh_contact_indentity: bool = False,
    dataset: str = "contactpose",
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
        unique_mesh_pths, visited = [], []
        for path in set(mesh_pths):
            file_name = os.path.basename(path)
            if file_name in obj_cache or file_name in visited:
                continue
            visited.append(file_name)
            unique_mesh_pths.append(path)
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
                    n_samples=n_samples,
                    n_normals=n_normals,
                    dataset=dataset,
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


def compute_contact_coverage(
    hand_verts: torch.Tensor,
    faces: torch.Tensor,
    obj_points: torch.Tensor,
    thresh_mm,
    base_unit: float,
    n_samples: int = 10000,
) -> torch.Tensor:
    contact_coverage = []
    for i in range(hand_verts.shape[0]):
        pred_hand_mesh = trimesh.Trimesh(
            vertices=hand_verts[i].detach().cpu().numpy(),
            faces=faces.detach().cpu().numpy(),
        )
        hand_points = to_cuda_(
            torch.from_numpy(
                trimesh.sample.sample_surface(pred_hand_mesh, n_samples)[0]
            ).float()
        )
        dists = torch.cdist(hand_points, obj_points[i].to(hand_points.device))  # (N, N)
        dists = dists.min(
            dim=1
        ).values  # (N): distance of each hand point to the closest object point
        contact_coverage.append(
            (dists <= (thresh_mm / base_unit)).sum() / n_samples * 100
        )
    return torch.stack(contact_coverage).mean().item()


def compute_binary_contacts(
    hand_verts: torch.Tensor,
    faces: torch.Tensor,
    obj_points: torch.Tensor,
    thresh_mm: float,
    base_unit: float,
    n_mesh_upsamples: int = 2,
    return_upsampled_verts: bool = True,
) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
    bs = hand_verts.shape[0]
    if n_mesh_upsamples > 0:
        if type(faces) is not list:
            faces = [faces] * bs
        upsample_layer = UpSampleLayer()
        for _ in range(n_mesh_upsamples):
            hand_verts, faces = upsample_layer(hand_verts, faces)
    dists = torch.cdist(
        hand_verts, obj_points.to(hand_verts.device)
    )  # (B, VERTS, N_OBJ_POINTS)
    dists = dists.min(dim=-1).values
    binary_contacts = (dists <= (thresh_mm / base_unit)).int()
    return (
        binary_contacts,
        hand_verts if return_upsampled_verts else None,
    )


def compute_contact_fidelity(
    canonical_verts: torch.Tensor,
    pred_bin_contacts: torch.Tensor,
    gt_bin_contacts: torch.Tensor,
    normalize_false_positives: bool = False,
) -> torch.Tensor:
    """
    Compute the contact fidelity between the predicted and ground-truth binary contacts, as a
    modified Hamming distance with a normalized L2-norm-based penalty term for false positives and
    a binary term corresponding to "Hamming score" for false negatives.
    """
    # Compute the true positives as the binary AND between the predicted and ground-truth binary contacts:
    true_positive_scores = (pred_bin_contacts & gt_bin_contacts).sum(dim=-1).float()
    # Compute the normalized L2-norm-based penalty term for false positives:
    false_positive_bool_indices = (gt_bin_contacts == 0) & (pred_bin_contacts == 1)
    dists = torch.cdist(canonical_verts, canonical_verts)  # (B, VERTS, VERTS)
    true_positive_indices = gt_bin_contacts == 1
    dists = torch.where(
        true_positive_indices[:, None, :], dists, torch.inf
    )  # (B, VERTS, VERTS). All GT vertices that aren't in contact have inf distance so they won't be selected.
    # This seems much slower than torch.where: dists = dists * (torch.ones_like(dists) + true_positive_indices[:, None, :].int() * torch.inf)
    nearest_true_contacts = dists.min(dim=-1).values  # (B, VERTS)
    normalizer = 1.0
    if normalize_false_positives:
        # Compute the normalizing constant for the false positive penalty term, as the cardinality of
        # the set of direct neighbors of the false positive vertices:
        raise NotImplementedError("This is not implemented yet.")
    # Now mask out the nearest true contact distances to keep only the false positive distances:
    false_positive_penalties = (
        torch.where(
            false_positive_bool_indices,
            torch.exp(nearest_true_contacts) * normalizer,
            torch.zeros_like(nearest_true_contacts),
        )
        .sum(dim=-1)
        .float()
    )
    return (true_positive_scores - false_positive_penalties).mean().cpu()
