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
import pickle
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import open3d.geometry as o3dg
import open3d.io as o3dio
import open3d.utility as o3du
import torch
import trimesh
import trimesh.voxel.creation as voxel_create
from manotorch.upsamplelayer import UpSampleLayer
from tqdm import tqdm

from utils import to_cuda_


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
        gt_obj_contacts = None  # TODO
    else:
        obj_mesh = o3dio.read_triangle_mesh(path)
        # From ContactOpt:
        vertex_colors = np.array(obj_mesh.vertex_colors, dtype=np.float32)
        gt_obj_contacts = torch.from_numpy(
            np.expand_dims(fit_sigmoid(vertex_colors[:, 0]), axis=1)
        )  # Normalize with sigmoid, shape (V, 1)
    if center_on_obj_com:
        obj_mesh.translate(-obj_mesh.get_center())
    #obj_points = torch.from_numpy(
    #    np.asarray(obj_mesh.sample_points_uniformly(n_samples).points)
    #).float()
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
    t_obj_mesh = trimesh.Trimesh(vertices=obj_mesh.vertices, faces=obj_mesh.triangles, process=False, validate=False).copy() # Don't remove my precious vertices you filthy animal!!!
    assert t_obj_mesh.vertices.shape[0] == np.asarray(obj_mesh.vertices).shape[0], "Trimesh changed the vert count!"
    assert (
        t_obj_mesh.vertices.shape[0] == gt_obj_contacts.shape[0]
    ), f"Object mesh has {t_obj_mesh.vertices.shape[0]} vertices and ground-truth object contacts have shape {gt_obj_contacts.shape}."
    del obj_mesh
    voxel = None
    if compute_iv:
        # TODO: Use cuda_voxelizer + libmesh's SDF-based check_mesh_contains
        voxel = (
            voxel_create.local_voxelize(t_obj_mesh, np.array([0, 0, 0]), pitch, radius)
            .fill()
            .matrix
        )
    obj_data = {
        "mesh": t_obj_mesh,
        "obj_contacts": gt_obj_contacts,
        #"points": obj_points,
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
    keep_mesh_contact_indentity: bool = True,
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
        keep_mesh_contact_indentity: Whether to treat all meshes as unique (they have their own
        contact maps) or reduce all identitical meshes to a single one. Must be True to compute the
        precision/recall and F1 scores!
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
    keep_mesh_contact_indentity: bool = True,
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
            to_cuda_(torch.stack(v, dim=0))
            if type(v[0]) == torch.Tensor and k != "obj_contacts"
            else v
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

     _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
    | This is very dumb... I was basically almost reinventing F1 score on binary contact maps!!!|
     _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

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


# This function is taken from the ContactPose repo:
def fit_sigmoid(colors, a=0.05):
    """Fits a sigmoid to raw contact temperature readings from the ContactPose dataset. This function is copied from that repo"""
    idx = colors > 0
    ci = colors[idx]

    x1 = min(ci)  # Find two points
    y1 = a
    x2 = max(ci)
    y2 = 1 - a

    lna = np.log((1 - y1) / y1)
    lnb = np.log((1 - y2) / y2)
    k = (lnb - lna) / (x1 - x2)
    mu = (x2 * lna - x1 * lnb) / (lna - lnb)
    ci = np.exp(k * (ci - mu)) / (1 + np.exp(k * (ci - mu)))  # Apply the sigmoid
    colors[idx] = ci
    return colors


def compute_contacts_fscore(
    pred_hand_mesh_w_obj_contacts_mesh: Tuple[
        trimesh.Trimesh, trimesh.Trimesh, torch.Tensor
    ],
    thresh_mm=2,
    base_unit_mm: int = 1000,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred_hand_mesh, obj_mesh, gt_obj_contacts = pred_hand_mesh_w_obj_contacts_mesh
    pred_hand_pts = (
        torch.from_numpy(trimesh.sample.sample_surface(pred_hand_mesh, 5000)[0])
        .float()
        .cpu()
    )
    # 1. Compute a binary *object* contact map by computing a boolean mask where object points are
    # within 2mm of hand points (sample points on meshes).
    assert (
        obj_mesh.vertices.shape[0] == gt_obj_contacts.shape[0]
    ), f"Object mesh has {obj_mesh.vertices.shape[0]} vertices and ground-truth object contacts have shape {gt_obj_contacts.shape}."
    obj_verts = (
        torch.from_numpy(np.asarray(obj_mesh.vertices)).float().to(pred_hand_pts.device)
    )
    pred_obj_contacts = torch.zeros(obj_verts.shape[0]).to(pred_hand_pts.device)
    dists = torch.cdist(obj_verts, pred_hand_pts)
    min_dists = torch.min(dists, dim=1)[0]
    pred_obj_contacts[min_dists < (thresh_mm / base_unit_mm)] = 1
    pred_obj_contacts = pred_obj_contacts.unsqueeze(-1)
    assert (
        pred_obj_contacts.shape == gt_obj_contacts.shape
    ), f"Predicted object contacts have shape {pred_obj_contacts.shape} and ground-truth object contacts have shape {gt_obj_contacts.shape}."
    # 2. Take the GT thermal contact map, threshold it at 0.4 to make it binary.
    gt_obj_contacts = (gt_obj_contacts > 0.4).int().to(pred_hand_pts.device)
    """ End of code taken from ContactOpt """
    assert gt_obj_contacts.sum() > 0, "The ground-truth contact map is empty!"
    # 3. Call calculate_fscore on the two binary maps!
    # Compute precision
    true_positives = (pred_obj_contacts * gt_obj_contacts).sum().cpu().float()
    predicted_positives = pred_obj_contacts.sum().cpu().float()
    precision = true_positives / predicted_positives if predicted_positives > 0 else 0

    # Compute recall
    actual_positives = gt_obj_contacts.sum().cpu().float()
    recall = true_positives / actual_positives if actual_positives > 0 else 0
    # Compute F1 score
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return f1_score * 100.0, precision * 100.0, recall * 100.0


def mp_compute_contacts_fscore(
    pred_hand_meshes: List[trimesh.Trimesh],
    obj_meshes: List[trimesh.Trimesh],
    obj_contacts: List[torch.Tensor],
    thresh_mm=2,
    base_unit_mm: int = 1000,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with multiprocessing.Pool(min(os.cpu_count() - 2, len(obj_meshes))) as pool:
        results = tqdm(
            pool.imap(
                partial(
                    compute_contacts_fscore,
                    thresh_mm=thresh_mm,
                    base_unit_mm=base_unit_mm,
                ),
                zip(
                    pred_hand_meshes,
                    obj_meshes,
                    [obj_contacts[i].cpu() for i in range(len(obj_contacts))],
                ),
            ),
            total=len(obj_meshes),
            desc="Computing F1 scores for predicted object contacts",
        )

        # The result is a list of tuples, and we want the mean of each element of the tuples:
        f1_scores, precisions, recalls = zip(*list(results))
        return (
            torch.tensor(f1_scores).mean(),
            torch.tensor(precisions).mean(),
            torch.tensor(recalls).mean(),
        )
