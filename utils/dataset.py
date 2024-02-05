#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Dataset-related functions.
"""

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import numpy as np
import open3d
import torch
from bps_torch.bps import bps_torch
from pytorch3d.structures import Pointclouds
from pytorch3d.transforms import Transform3d, random_rotation
from pytorch3d.transforms.rotation_conversions import (
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)
from scipy.spatial import cKDTree

from utils import to_cuda


def transform_verts(
    verts: torch.Tensor, rot6d: torch.Tensor, t: torch.Tensor, apply_inverse_rot: bool
) -> torch.Tensor:
    R = rotation_6d_to_matrix(rot6d)
    if apply_inverse_rot:
        # NOTE: This is the inverse of the transform we want to apply to the MANO model. It works
        # for the code given in ContactPose because they're applying the inverse transform to the
        # points with np.dot(transform, points.T).
        inv_r = Transform3d().rotate(R).inverse().to(verts.device)
        transform = (
            Transform3d(device=verts.device)
            .compose(inv_r)
            .translate(t)
            .to(verts.device)
        )
    else:
        transform = (
            Transform3d(device=verts.device).rotate(R).translate(t).to(verts.device)
        )
    return transform.transform_points(verts)


def compute_hand_object_pair_scalar(
    hand_anchors: torch.Tensor,
    object_point_cloud: torch.Tensor,
    scale_by_object_only: bool = False,
) -> torch.Tensor:
    # Build a pointcloud from the object points + hand anchors with open3D/Pytorch3D.
    pointcloud = Pointclouds(
        (
            torch.cat([object_point_cloud, hand_anchors.squeeze(0).cpu()], dim=0)
            if not scale_by_object_only
            else object_point_cloud
        ).unsqueeze(0)
    )
    # Now scale the pointcloud to unit sphere. First, find the scalar to apply. We can
    # get the bounding boxes and find the min and max of the diagonal of the bounding
    # boxes, and then compute each scalar such that the diagonal of the bounding box is 1.

    # (N, 3, 2) where bbox[i,j] gives the min and max values of mesh i along axis j. So
    # bbox[i,:,0] is the min values of mesh i along all axes, and bbox[i,:,1] is the max
    # values of mesh i along all axes.
    bboxes = pointcloud.get_bounding_boxes()
    bboxes_diag = torch.norm(bboxes[:, :, 1] - bboxes[:, :, 0], dim=1)  # (N,) but N=1
    hand_object_scalar = 1.0 / bboxes_diag  # (N,)
    # Make sure that the new bounding boxes have a diagonal of 1.
    rescaled_pointcloud = pointcloud.scale(hand_object_scalar.unsqueeze(1))  # (N, 3, M)
    bboxes = rescaled_pointcloud.get_bounding_boxes()
    bboxes_diag = torch.norm(bboxes[:, :, 1] - bboxes[:, :, 0], dim=1)
    assert torch.allclose(
        bboxes_diag, torch.ones_like(bboxes_diag)
    ), "Bounding boxes are not unit cubes."
    return hand_object_scalar.float()


@to_cuda
def compute_choir(
    pointcloud: torch.Tensor,
    anchors: torch.Tensor,
    scalar: float,
    bps: torch.Tensor,
    anchor_indices: torch.Tensor,
    remap_bps_distances: bool,
    exponential_map_w: float,
    use_deltas: bool,
    compute_hand_object_distances: Optional[bool] = False,
) -> torch.Tensor:
    """
    Args:
        pointcloud: Shape (B, N, 3)
        anchors: Shape (B, N_ANCHORS, 3)
        scalar (float): Scalar for the hand and object pointcloud such that they end up in the unit sphere.
        bps: Shape (B, N_BPS_POINTS, 3). It is important that we reuse the same BPS!
        remap_bps_distances: If True, remap the BPS distances to [0, 1] using the exponential map from the GOAL paper.
        exponential_map_w: The w parameter for the exponential map: exp(-w * d).
    """
    assert len(anchors.shape) == 3
    assert len(pointcloud.shape) == 3
    assert len(bps.shape) == 3
    B = anchors.shape[0]
    bps_encoder = bps_torch(custom_basis=bps)
    # TODO: Investigate why parts of the pointcloud (i.e. in the wine glass) are ignored during
    # sampling, especially when reducing the dimensionality of the BPS representation. Well this is
    # due to how it works and there's not much we can do about it except rescaling it properly, but
    # that won't do much.
    rescaled_obj_pointcloud = pointcloud * scalar
    rescaled_anchors = anchors * scalar
    object_bps: Dict[str, Any] = bps_encoder.encode(
        rescaled_obj_pointcloud,
        feature_type=["dists", "deltas"],
    )
    object_bps["deltas"] = object_bps["deltas"].to(bps.device)
    object_bps["dists"] = object_bps["dists"].to(bps.device)
    rescaled_ref_pts = bps + object_bps["deltas"]  # A subset of rescaled_obj_pointcloud
    # Assign all ordered 32 rescaled_anchors to a batch of BPS points and repeat for all available
    # batches. The number of batches is determined by the number of BPS points, and the latter
    # must be a multiple of 32.
    assert (
        bps.shape[1] % 32 == 0
    ), f"The number of BPS points ({bps.shape[1]}) must be a multiple of 32."
    anchor_ids = anchor_indices.unsqueeze(0).repeat((B, 1))
    if use_deltas:
        assert (
            not remap_bps_distances
        ), "Remapping BPS distances is not supported with deltas."
        anchor_deltas = bps[:, :, None, :] - rescaled_anchors[:, None, :, :]
        anchor_deltas = torch.gather(
            anchor_deltas, 2, anchor_ids[..., None, None].repeat(1, 1, 1, 3)
        ).squeeze(-2)
        if compute_hand_object_distances:
            raise NotImplementedError
        choir = torch.cat(
            (
                object_bps["deltas"],
                anchor_deltas,
            ),
            dim=-1,
        )
    else:
        # Compute the distances between the BPS points and the MANO anchors:
        anchor_distances = torch.cdist(
            bps, rescaled_anchors
        )  # Shape: (BPS_LEN, N_ANCHORS)
        anchor_distances = torch.gather(
            anchor_distances, 2, anchor_ids.unsqueeze(-1)
        ).squeeze(-1)
        if compute_hand_object_distances:
            raise NotImplementedError
            # Compute the distances between the BPS points and the MANO anchors:
            ho_dist = torch.cdist(
                rescaled_ref_pts, rescaled_anchors
            )  # Shape: (BPS_LEN, N_ANCHORS)
            ho_dist = torch.gather(ho_dist, 2, anchor_ids.unsqueeze(-1)).squeeze(-1)
        choir = torch.cat(
            [
                object_bps["dists"]
                .unsqueeze(-1)
                .repeat(
                    (B, 1, 1)
                ),  # O(len(BPS)) because the anchor distances don't involve KNN.
                anchor_distances.unsqueeze(-1),
            ],
            dim=-1,
        )
        if remap_bps_distances:
            # Remap the BPS distances to [0, 1] using the exponential map from the GOAL paper. 1 is
            # very close to the BPS point, 0 is very far away.
            choir = torch.exp(-exponential_map_w * choir)
    return choir, rescaled_ref_pts  # , anchor_deltas


def compute_hand_contacts_simple(
    obj_pointcloud: torch.Tensor,
    hand_verts: torch.Tensor,
    tolerance_mm: float = 10,
) -> torch.Tensor:
    """
    Compute the hand contacts as a vector of normalized contact values in [0, 1] for each of the 778 MANO vertices.
    This approach is very crude because even points that are tolerance_mm away LATERALLY from the hand
    vertex are considered to be in contact. I should probably approach the capsule idea of
    ContactOpt or something cheaper but better (normal vectors?).
    """
    # Go through each MANO vertex and compute the distance to the object pointcloud (nearest
    # neighbor). Do this in a vectorized fashion.
    # TODO: Get the signed distance instead of the absolute distance! Or a tolerance of
    # -tolerance_mm cropped to 0 but that's kind of the same thing.
    distances = torch.cdist(hand_verts, obj_pointcloud)  # Shape: (778, N_OBJ_POINTS)
    # Find the minimum distance for each MANO vertex:
    distances = (
        distances.min(dim=-1).values * 1000
    )  # Shape: (778,). The scaling is to convert to millimeters.
    # If the distance is within tolerance_mm, then use the distance as the contact value, otherwise
    # use 0. Use signed distances to allow for contact on both sides of the object (penetration due
    # to the non-representation of soft tissue).
    distances[distances > tolerance_mm] = 0.0
    # Normalize the contact values to [0, 1] (i.e. divide by tolerance_mm).
    distances_exp = torch.exp(distances / tolerance_mm)
    distances = (
        distances_exp / distances_exp.max()
    )  # We now have 1 for least contact and 0 for max contact. It's the opposite of what we want!
    return 1.0 - distances


def get_contact_counts_by_neighbourhoods(
    mano_vertices: torch.Tensor,
    mano_normals: torch.Tensor,
    object_points: torch.Tensor,
    tolerance_cone_angle: int,
    tolerance_mm: int,
    base_unit: float,
    K=100,
) -> torch.Tensor:
    """
    Compute the contact counters for each MANO vertex. They are computed by counting the number of
    object points within a cone around each MANO vertex in the direction of the vertex normal.

    Args:
        mano_vertices: A tensor of shape (N, 3) representing the N MANO vertices.
        mano_normals: A tensor of shape (N, 3) representing the normals at the N MANO vertices.
        object_points: A tensor of shape (P, 3) representing the P object points (sampled uniformly).
        tolerance_cone_angle: The angle in degrees of the cone within which to count object points.
        tolerance_mm: The distance in millimeters within which to count object points.
        base_unit: The number of mm per unit in the dataset.
        K: The number of nearest object points to consider when computing the contact counters.
    Returns:
        A tensor of shape (N,) representing the raw contact counters for all N MANO vertices.
    """

    # Convert mano_vertices and object_points to numpy arrays
    mano_vertices = np.array(mano_vertices)
    object_points = np.array(object_points)

    # ========= Compute the contact counters for each MANO vertex =========
    # Build a KDTree for object_points
    object_tree = cKDTree(object_points)

    # Initialize contact counters
    contact_counters = np.zeros(len(mano_vertices))

    # Iterate over each MANO vertex
    print(f"MANO vertices shape: {mano_vertices.shape}")
    print(f"Object points shape: {object_points.shape}")
    for i, vertex in enumerate(mano_vertices):
        # Get the normal vector for the current MANO vertex
        normal = mano_normals[i]

        # Find the K nearest neighbors of the current MANO vertex
        _, indices = object_tree.query(vertex, k=K)
        neighbors = object_points[indices]
        print(f"Computing contact counters for vertex {i}: {vertex}")
        print(f"Nearest neighbors: {neighbors.shape}")

        # Compute the distances between the MANO vertex and its K nearest neighbors
        to_obj_vectors = neighbors - vertex
        print(f"To object vectors: {to_obj_vectors.shape}")
        distances = np.linalg.norm(to_obj_vectors, axis=1)
        print(
            f"Distances: {distances.shape}. Range (mm): {np.min(distances) * base_unit} - {np.max(distances) * base_unit}"
        )

        # Compute the angles between the normal vector and the vectors from the MANO vertex to its K nearest neighbors
        angles = np.arccos(
            np.dot(normal, to_obj_vectors.T) / (np.linalg.norm(normal) * distances)
        )
        print(f"Angles: {angles.shape}. Range: {np.min(angles)} - {np.max(angles)}")

        # Count the number of neighbors within the cone of tolerance
        contact_counters[i] = np.sum(
            (angles <= tolerance_cone_angle) & ((distances * base_unit) <= tolerance_mm)
        )

    print("=========================================")
    print(f"Contact counters: {contact_counters.shape}")
    print(f"Range: {np.min(contact_counters)} - {np.max(contact_counters)}")
    print("=========================================")
    return torch.from_numpy(contact_counters)


def compute_anchor_gaussians(
    mano_vertices: torch.Tensor,
    mano_anchors: torch.Tensor,
    contact_counters: torch.Tensor,
    base_unit: float,
    anchor_mean_threshold_mm: float,
    min_contact_points_for_neighbourhood: int,
    debug_anchor: Optional[int] = None,
) -> torch.Tensor:
    """
    Fit a 3D Gaussian on each anchor neighbourhood of MANO vertices, with given contact counters.
    This is done by multiplying the vertices by their contact counters and then computing the mean
    and covariance matrix of the clusters.

    Args:
        mano_vertices: A tensor of shape (N, 3) representing the N MANO vertices.
        mano_anchors: A tensor of shape (N_ANCHORS, 3) representing the N_ANCHORS anchor points.
        contact_counters: A tensor of shape (N,) representing the raw contact counters for all N MANO vertices.
        base_unit: The number of mm per unit in the dataset.
        anchor_mean_threshold_mm: The max distance (mm) between the mean of the cluster and the anchor point.
        min_contact_points_for_neighbourhood: The minimum number of contact points required to consider a neighbourhood.
        debug_anchor: If not None, only compute the contact values for the anchor with the given index.
    Returns:
        A tensor of shape (N_ANCHORS, 3, 3, 3) representing the 3D Gaussian parameters (mean and
        covariance matrix) for each anchor neighbourhood.
    """

    # ================================================================
    # ======= Normalize the contact counters for each anchor neighborhood =========
    gaussian_params = torch.zeros((mano_anchors.shape[0], 12))
    # Get distances to anchors
    anchor_distances = torch.cdist(
        torch.tensor(mano_vertices), mano_anchors
    )  # Shape: (778, N_ANCHORS)
    print(f"Anchor shapes: {mano_anchors.shape}")
    print(f"Anchor distances: {anchor_distances.shape}")
    # Keep nearest anchor index for each vertex
    anchor_indices = torch.topk(anchor_distances, 1, largest=False).indices  # (778, 1)
    print(
        f"Nearest anchor indices: {anchor_indices.shape}. Range: {torch.min(anchor_indices)} - {torch.max(anchor_indices)}"
    )
    anchor_contacts = {}
    for anchor in mano_anchors:
        # Get the indices of MANO vertices belonging to the current neighborhood
        anchor_id = int(torch.where(mano_anchors == anchor)[0][0])
        if debug_anchor is not None and anchor_id != debug_anchor:
            continue
        print(f"Computing contact values for anchor {anchor_id}.")
        neighbour_vert_indices = torch.where(anchor_indices == anchor_id)[
            0
        ]  # List of indices of vertices belonging to the current anchor neighborhood
        print(
            f"Neighbour vertex indices: {neighbour_vert_indices.shape}. Range: {torch.min(neighbour_vert_indices)} - {torch.max(neighbour_vert_indices)}"
        )
        print(
            f"Contact counters: {contact_counters[neighbour_vert_indices].shape}. Range: {torch.min(contact_counters[neighbour_vert_indices])} - {torch.max(contact_counters[neighbour_vert_indices])}"
        )
        print(
            f"Non-zero contact counters: {torch.where(contact_counters[neighbour_vert_indices] > 0, 1.0, 0.0).sum()}"
        )
        print(
            f"MANO vertices in neighbourhood: {mano_vertices[neighbour_vert_indices].shape}"
        )
        if (
            torch.where(contact_counters[neighbour_vert_indices] > 0, 1.0, 0.0).sum()
            < min_contact_points_for_neighbourhood
        ):
            gaussian_params[anchor_id] = torch.zeros(12)
            anchor_contacts[anchor_id] = (
                mano_vertices[neighbour_vert_indices],
                mano_vertices[neighbour_vert_indices],
                contact_counters[neighbour_vert_indices],
            )
            continue
        # Duplicate vertices by their contact counters
        # First, remove neighbour_vert_indices where the contact counter is 0:
        neighbour_vert_indices = neighbour_vert_indices[
            contact_counters[neighbour_vert_indices] > 0
        ]
        cluster_points = torch.repeat_interleave(
            mano_vertices[neighbour_vert_indices],
            contact_counters[neighbour_vert_indices].int(),
            # + torch.ones_like(contact_counters[neighbour_vert_indices].int()),
            dim=0,
        )
        anchor_contacts[anchor_id] = (
            mano_vertices[neighbour_vert_indices],
            cluster_points,
            contact_counters[neighbour_vert_indices],
        )
        print(f"Cluster points: {cluster_points.shape}.")
        mean = torch.mean(cluster_points, dim=0)
        print(f"L2_Norm(mean-anchor) (mm): {torch.norm(mean - anchor) * base_unit}")
        if torch.norm(mean - anchor) * base_unit > anchor_mean_threshold_mm:
            print(
                f"Mean of cluster is too far from anchor: {torch.norm(mean - anchor) * base_unit} > {anchor_mean_threshold_mm}"
            )
            gaussian_params[anchor_id] = torch.zeros(12)
        else:
            cov = torch.cov((cluster_points - mean).T)
            print(f"Mean: {mean.shape}. Cov: {cov.shape}")
            # /!\ The mean must be shifted to the origin, so that we can shift it back in TTO!
            mean = mean - anchor
            gaussian_params[anchor_id] = torch.cat((mean, cov.flatten()))

    return gaussian_params, anchor_contacts


def compute_hand_contacts_bps(
    obj_pointcloud: torch.Tensor,
    hand_pointcloud: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:
    """
    Compute the hand contacts and encode it all in a BPS representation.
    """
    raise NotImplementedError


@to_cuda
def pack_and_pad_sample_label(
    theta,
    beta,
    rot_6d,
    trans,
    choir,
    rescaled_ref_pts,
    scalar,
    hand_idx,
    gt_choir,
    gt_rescaled_ref_pts,
    gt_scalar,
    gt_joints,
    gt_anchors,
    gt_theta,
    gt_beta,
    gt_rot_6d,
    gt_trans,
    bps_dim,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Why do I do this? I was convinced that it would be faster to collate the samples and labels
    into a batch when using multiple workers because of huge slowdowns I experienced when I was
    using Dicts. Now I'm not sure it was the real culprit but this helps a bit. Judge me, I don't
    care.
    """
    # ========= Without Pytorch 2.0 Nested Tensor (annoying but I use old clusters) =========
    max_sample_dim = max(
        [
            s.shape[-1]
            for s in set().union(
                theta,
                beta,
                rot_6d,
                trans,
                rescaled_ref_pts,
                choir,
            )
        ]
    )
    max_label_dim = max(
        [
            s.shape[-1]
            for s in set().union(
                gt_choir,
                gt_rescaled_ref_pts,
                gt_joints,
                gt_anchors,
                gt_theta,
                gt_beta,
                gt_rot_6d,
                gt_trans,
            )
        ]
    )
    sample = torch.stack(
        [
            torch.nn.functional.pad(
                choir.squeeze(0), (0, max_sample_dim - choir.shape[-1]), value=0.0
            ),
            torch.nn.functional.pad(
                rescaled_ref_pts.squeeze(0),
                (0, max_sample_dim - rescaled_ref_pts.shape[-1]),
                value=0.0,
            ),
            torch.nn.functional.pad(
                scalar.unsqueeze(0),
                (0, max_sample_dim - scalar.unsqueeze(0).shape[-1]),
                value=0.0,
            ).repeat((bps_dim, 1)),
            torch.nn.functional.pad(
                (torch.ones((1, 1)) if hand_idx == "right" else torch.zeros((1, 1))),
                (0, max_sample_dim - 1),
                value=0.0,
            ).repeat((bps_dim, 1)),
            torch.nn.functional.pad(
                theta, (0, max_sample_dim - theta.shape[-1]), value=0.0
            ).repeat((bps_dim, 1)),
            torch.nn.functional.pad(
                beta, (0, max_sample_dim - beta.shape[-1]), value=0.0
            ).repeat((bps_dim, 1)),
            torch.nn.functional.pad(
                rot_6d, (0, max_sample_dim - rot_6d.shape[-1]), value=0.0
            ).repeat((bps_dim, 1)),
            torch.nn.functional.pad(
                trans, (0, max_sample_dim - trans.shape[-1]), value=0.0
            ).repeat((bps_dim, 1)),
        ],
        dim=0,
    ).cpu()

    padded_joints = torch.zeros((bps_dim, max_label_dim), device=gt_joints.device)
    padded_joints[: gt_joints.squeeze(0).shape[0], :3] = gt_joints.squeeze(0)
    padded_anchors = torch.zeros((bps_dim, max_label_dim), device=gt_anchors.device)
    padded_anchors[: gt_anchors.squeeze(0).shape[0], :3] = gt_anchors.squeeze(0)
    label = torch.stack(
        [
            torch.nn.functional.pad(
                gt_choir.squeeze(0),
                (0, max_label_dim - gt_choir.shape[-1]),
                value=0.0,
            ),
            torch.nn.functional.pad(
                gt_rescaled_ref_pts.squeeze(0),
                (0, max_label_dim - gt_rescaled_ref_pts.shape[-1]),
                value=0.0,
            ),
            torch.nn.functional.pad(
                gt_scalar.unsqueeze(0),
                (0, max_label_dim - gt_scalar.unsqueeze(0).shape[-1]),
                value=0.0,
            ).repeat((bps_dim, 1)),
            padded_joints,
            padded_anchors,
            gt_theta.repeat((bps_dim, 1)),
            torch.nn.functional.pad(
                gt_beta, (0, max_label_dim - gt_beta.shape[-1]), value=0.0
            ).repeat((bps_dim, 1)),
            torch.nn.functional.pad(
                gt_rot_6d, (0, max_label_dim - gt_rot_6d.shape[-1]), value=0.0
            ).repeat((bps_dim, 1)),
            torch.nn.functional.pad(
                gt_trans, (0, max_label_dim - gt_trans.shape[-1]), value=0.0
            ).repeat((bps_dim, 1)),
        ],
        dim=0,
    ).cpu()
    if "2.0" in torch.__version__:
        # ============ With Pytorch 2.0 Nested Tensor ============
        gt_sample = torch.nested.to_padded_tensor(
            torch.nested.nested_tensor(
                [
                    choir.squeeze(0),  # (N, 2)
                    rescaled_ref_pts.squeeze(0),  # (N, 3)
                    scalar.unsqueeze(0),  # (1, 1)
                    torch.ones((1, 1))
                    if hand_idx == "right"
                    else torch.zeros((1, 1)),  # (1, 1)
                    theta,  # (1, 18)
                    beta,  # (1, 10)
                    rot_6d,  # (1, 6)
                    trans,  # (1, 3)
                ]
            ),
            0.0,
        ).cpu()

        gt_label = torch.nested.to_padded_tensor(
            torch.nested.nested_tensor(
                [
                    gt_choir.squeeze(0),  # (N, 2)
                    gt_rescaled_ref_pts.squeeze(0),  # (N, 3)
                    gt_scalar.unsqueeze(0),  # (1, 1)
                    gt_joints.squeeze(0),  # (21, 3)
                    gt_anchors.squeeze(0),  # (32, 3)
                    gt_theta,  # (1, 18)
                    gt_beta,  # (1, 10)
                    gt_rot_6d,  # (1, 6)
                    gt_trans,  # (1, 3)
                ]
            ),
            0.0,
        ).cpu()
        # ============================================================
        # ============== Test that manual padding works ==============
        assert torch.allclose(gt_sample[0, :], sample[0, :]), "CHOIR mismatch."
        assert torch.allclose(gt_sample[1, :], sample[1, :]), "Ref pts mismatch."
        assert torch.allclose(gt_sample[2, 0], sample[2, 0]), "Scalar mismatch."
        assert torch.allclose(gt_sample[3, 0], sample[3, 0]), "Hand idx mismatch."
        assert torch.allclose(gt_sample[4, 0], sample[4, 0]), "Theta mismatch."
        assert torch.allclose(gt_sample[5, 0], sample[5, 0]), "Beta mismatch."
        assert torch.allclose(gt_sample[6, 0], sample[6, 0]), "Rot mismatch."
        assert torch.allclose(gt_sample[7, 0], sample[7, 0]), "Trans mismatch."

        assert torch.allclose(gt_label[0, :], label[0, :]), "CHOIR mismatch."
        assert torch.allclose(gt_label[1, :], label[1, :]), "Ref pts mismatch."
        assert torch.allclose(gt_label[2, 0], label[2, 0]), "Scalar mismatch."
        assert torch.allclose(gt_label[3, :21], label[3, :21]), "Joints mismatch."
        assert torch.allclose(gt_label[4, :32], label[4, :32]), "Anchors mismatch."
        assert torch.allclose(gt_label[5, 0], label[5, 0]), "Theta mismatch."
        assert torch.allclose(gt_label[6, 0], label[6, 0]), "Beta mismatch."
        assert torch.allclose(gt_label[7, 0], label[7, 0]), "Rot mismatch."
        assert torch.allclose(gt_label[8, 0], label[8, 0]), "Trans mismatch."
        # =============================================================
    return sample, label


def get_scalar(anchors, obj_ptcld, scaling) -> torch.Tensor:
    if scaling == "pair":
        scalar = compute_hand_object_pair_scalar(deepcopy(anchors), deepcopy(obj_ptcld))
    elif scaling == "object":
        scalar = compute_hand_object_pair_scalar(
            deepcopy(anchors), deepcopy(obj_ptcld), scale_by_object_only=True
        )
    elif scaling == "fixed":
        scalar = torch.tensor([10.0])
    elif scaling == "none":
        scalar = torch.tensor([1.0])
    else:
        raise ValueError(f"Unknown rescale type {scaling}")
    return scalar


def augment_hand_object_pose(
    obj_mesh: open3d.geometry.TriangleMesh, hTm: torch.Tensor, around_z: bool = True
) -> None:
    """
    Augment the object mesh with a random rotation and translation.
    """
    # Randomly rotate the object and hand meshes around the z-axis only:
    if around_z:
        R = open3d.geometry.get_rotation_matrix_from_xyz(
            np.array([0, 0, np.random.uniform(0, 2 * np.pi)])
        )
    else:
        R = random_rotation().cpu().numpy()
    # It is CRUCIAL to translate both to the center of the object before rotating, because the hand joints
    # are expressed w.r.t. the object center. Otherwise, weird things happen.
    rotate_origin = obj_mesh.get_center()
    obj_mesh.translate(-rotate_origin)
    # Rotate the object and hand
    obj_mesh.rotate(R, np.array([0, 0, 0]))
    r_hTm = torch.eye(4)
    # We need to rotate the 4x4 MANO root pose as well, by first converting R to a
    # 4x4 homogeneous matrix so we can apply it to the 4x4 pose matrix:
    R4 = torch.eye(4)
    R4[:3, :3] = torch.from_numpy(R).float()
    hTm[:, :3, 3] -= torch.from_numpy(rotate_origin).to(hTm.device, dtype=hTm.dtype)
    r_hTm = R4.to(hTm.device) @ hTm
    hTm = r_hTm
    return obj_mesh, hTm


def augment_hand_object_pose_grab(
    obj_mesh: open3d.geometry.TriangleMesh,
    params: Dict[str, torch.Tensor],
    use_affine_mano: bool,
    around_z: bool = True,
) -> None:
    """
    Augment the object mesh with a random rotation and translation.
    """
    # Randomly rotate the object and hand meshes around the z-axis only:
    if around_z:
        R = open3d.geometry.get_rotation_matrix_from_xyz(
            np.array([0, 0, np.random.uniform(0, 2 * np.pi)])
        )
    else:
        R = random_rotation().cpu().numpy()
    # It is CRUCIAL to translate both to the center of the object before rotating, because the hand joints
    # are expressed w.r.t. the object center. Otherwise, weird things happen.
    rotate_origin = obj_mesh.get_center()
    obj_mesh.translate(-rotate_origin)
    # Rotate the object and hand
    obj_mesh.rotate(R, np.array([0, 0, 0]))
    # I'm not sure why I need to invert the rotation here but it works. I could figure it out if I
    # had extra time but I don't. It might just be because the "rot" parameters are already the
    # inverse of the rotation applied to the object mesh when the hand is brought to the object
    # frame...
    rot_transform = Transform3d().rotate(torch.from_numpy(R).float()).inverse()
    if use_affine_mano:
        transform = (
            Transform3d()
            .rotate(rotation_6d_to_matrix(params["rot"]))
            .translate(params["trans"])
        )
        full_transform = Transform3d().compose(transform, rot_transform)
        params["trans"] = full_transform.get_matrix()[:, 3, :3]
        params["rot"] = matrix_to_rotation_6d(full_transform.get_matrix()[:, :3, :3])
    else:
        # TODO: Fix this. I prefer to stick with AffineMANO for now since it's in line with
        # ContactPose experiments and everything is simpler. Plus, TTO on a continuous 6D
        # rotation might be easier than for axis-angle but maybe not. But if I add a MANO
        # prediction objective to the model then yeah it'd be an advantage to use rot6D.
        raise NotImplementedError("SMPL-X parameters augmentation not working")
    return obj_mesh, params


def drop_fingertip_joints(joints: torch.Tensor, definition="snap") -> torch.Tensor:
    """
    Drop the fingertip joints from the input joint tensor.
    """
    if definition == "snap":
        if len(joints.shape) == 3:
            return joints[:, [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]]
        elif len(joints.shape) == 2:
            return joints[[0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]]
        else:
            raise ValueError(f"Unknown joint shape {joints.shape}")
    elif definition == "mano":
        if len(joints.shape) == 3:
            return joints[:, :16]
        elif len(joints.shape) == 2:
            return joints[:16]
        else:
            raise ValueError(f"Unknown joint shape {joints.shape}")


def snap_to_original_mano(snap_joints: torch.Tensor) -> torch.Tensor:
    """
    I'm using manotorch which returns joints in the SNAP definition:
    https://github.com/lixiny/manotorch/blob/5738d327a343e7533ad60da64d1629cedb5ae9e7/manotorch/manolayer.py#L240:
        # ** original MANO joint order (right hand)
        #                16-15-14-13-\
        #                             \
        #          17 --3 --2 --1------0
        #        18 --6 --5 --4-------/
        #        19 -12 -11 --10-----/
        #          20 --9 --8 --7---/

        # Reorder joints to match SNAP definition
        joints = joints[:, [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]]
    So if I have joints coming out of SMPL-X MANO, I need to reorder the manotorch joints to match.
    """
    if len(snap_joints.shape) == 3:
        return snap_joints[
            :,
            [0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3, 4, 8, 12, 16, 20],
        ]
    elif len(snap_joints.shape) == 2:
        return snap_joints[
            [0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3, 4, 8, 12, 16, 20]
        ]
    else:
        raise ValueError(f"Unknown shape {snap_joints.shape}")
