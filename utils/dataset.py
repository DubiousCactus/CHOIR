#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Dataset-related functions.
"""

from typing import Any, Dict, Optional, Tuple

import torch
from bps_torch.bps import bps_torch
from bps_torch.tools import normalize
from pytorch3d.transforms import Transform3d
from pytorch3d.transforms.rotation_conversions import rotation_6d_to_matrix


def transform_verts(
    verts: torch.Tensor, rot6d: torch.Tensor, t: torch.Tensor
) -> torch.Tensor:
    R = rotation_6d_to_matrix(rot6d)
    # NOTE: This is the inverse of the transform we want to apply to the MANO model. It works
    # for the code given in ContactPose because they're applying the inverse transform to the
    # points with np.dot(transform, points.T).
    inv_r = Transform3d().rotate(R).inverse().to(verts.device)
    transform = (
        Transform3d(device=verts.device).compose(inv_r).translate(t).to(verts.device)
    )
    return transform.transform_points(verts)


def compute_choir(
    pointcloud: torch.Tensor,
    anchors: torch.Tensor,
    pointclouds_mean: Optional[torch.Tensor] = None,
    bps_dim: int = 1024,
    anchor_assignment="random",
) -> Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:
    """
    For each BPS point, get the reference object point and compute the distance to the
    nearest MANO anchor. Append the anchor index to the BPS point value as well as the
    distance to the anchor: [BPS dists, BPS deltas, distance, one_hot_anchor_id]
    Note that we need the deltas to be able to fit MANO, since we'll need to reconstruct the
    pointcloud to compute the anchor distances!

    Args:
        pointcloud: Shape (N, 3)
        anchors: Shape (N_ANCHORS, 3)
        anchor_assignment: How to assign anchors to BPS points. Must be one of: "random",
        "closest", "closest_and_farthest", "batched_closest_and_farthest".
    """
    bps = bps_torch(
        bps_type="random_uniform",
        n_bps_points=bps_dim,
        radius=1.0,
        n_dims=3,
        custom_basis=None,
    )
    # TODO: Investigate why parts of the pointcloud (i.e. in the wine glass) are ignored during
    # sampling, especially when reducing the dimensionality of the BPS representation.
    normalized_pointcloud, pcl_mean, pcl_scalar = normalize(
        pointcloud.unsqueeze(0),
        x_mean=pointclouds_mean.unsqueeze(0) if pointclouds_mean is not None else None,
    )
    bps_enc: Dict[str, Any] = bps.encode(
        normalized_pointcloud,
        feature_type=["dists", "deltas"],
        x_features=None,
        custom_basis=None,
    )
    ref_ids = bps_enc["ids"][0]
    ref_pts = pointcloud[ref_ids]
    assert pointcloud[ref_ids[0], :].allclose(
        ref_pts[0, :]
    ), "ref_pts[0, :] != pointcloud[ref_ids[0], :]"
    # Compute the distances between the reference points and the anchors:
    anchor_distances = torch.cdist(
        ref_pts.float(), anchors.float()
    )  # Shape: (B, BPS_LEN, N_ANCHORS)
    anchor_encodings = []
    if anchor_assignment == "random":
        # Randomly sample an anchor for each reference point:
        anchor_ids = torch.randint(
            0,
            anchors.shape[1],
            (
                1,
                ref_pts.shape[0],
            ),
            device=ref_pts.device,
        )
        distances = torch.gather(anchor_distances, 2, anchor_ids.unsqueeze(-1)).squeeze(
            -1
        )
        anchor_encodings = [
            distances.unsqueeze(-1),
            torch.nn.functional.one_hot(anchor_ids, num_classes=anchors.shape[1]),
        ]
    elif anchor_assignment == "closest":
        anchor_ids = torch.argmin(anchor_distances, dim=2)
        distances = torch.gather(anchor_distances, 2, anchor_ids.unsqueeze(-1)).squeeze(
            -1
        )
        anchor_encodings = [
            distances.unsqueeze(-1),
            torch.nn.functional.one_hot(anchor_ids, num_classes=anchors.shape[1]),
        ]
    elif anchor_assignment == "closest_and_farthest":
        closest_anchor_ids = torch.argmin(anchor_distances, dim=2)
        closest_distances = torch.gather(
            anchor_distances, 2, closest_anchor_ids.unsqueeze(-1)
        ).squeeze(-1)
        farthest_anchor_ids = torch.argmax(anchor_distances, dim=2)
        farthest_distances = torch.gather(
            anchor_distances, 2, farthest_anchor_ids.unsqueeze(-1)
        ).squeeze(-1)
        anchor_encodings = [
            closest_distances.unsqueeze(-1),
            torch.nn.functional.one_hot(
                closest_anchor_ids, num_classes=anchors.shape[1]
            ),
            farthest_distances.unsqueeze(-1),
            torch.nn.functional.one_hot(
                farthest_anchor_ids, num_classes=anchors.shape[1]
            ),
        ]
    elif anchor_assignment == "batched_closest_and_farthest":
        raise NotImplementedError
    # Build the CHOIR representation:
    choir = torch.cat(
        [
            bps_enc["dists"].unsqueeze(-1),
            bps_enc["deltas"],
            *anchor_encodings,
        ],
        dim=-1,
    )
    return choir, pcl_mean, pcl_scalar, ref_pts  # for debugging


def compute_hand_contacts_simple(
    obj_pointcloud: torch.Tensor,
    hand_verts: torch.Tensor,
    tolerance_mm: float = 10,
) -> torch.Tensor:
    """
    Compute the hand contacts as a vector of normalized contact values in [0, 1] for each of the 778 MANO vertices.
    This approach is very crude because even points that are tolerance_mm away LATERALLY from the hand
    vertex are considered to be in contact. I should probably approach the capsule idea of
    ContactOpt or something cheaper but better.
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


def compute_hand_contacts_bps(
    obj_pointcloud: torch.Tensor,
    hand_pointcloud: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:
    """
    Compute the hand contacts and encode it all in a BPS representation.
    """
    raise NotImplementedError
