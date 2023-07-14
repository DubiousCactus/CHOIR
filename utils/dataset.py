#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Dataset-related functions.
"""

from typing import Any, Dict, Tuple

import torch
from bps_torch.bps import bps_torch
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
    scalar: float,
    bps: torch.Tensor,
    anchor_assignment="random",
) -> torch.Tensor:
    """
    For each BPS point, get the reference object point and compute the distance to the
    nearest MANO anchor. Append the anchor index to the BPS point value as well as the
    distance to the anchor: [BPS dists, BPS deltas, distance, one_hot_anchor_id]
    Note that we need the deltas to be able to fit MANO, since we'll need to reconstruct the
    pointcloud to compute the anchor distances!

    Args:
        pointcloud: Shape (B, N, 3)
        anchors: Shape (B, N_ANCHORS, 3)
        scalar (float): Scalar for the hand and object pointcloud such that they end up in the unit sphere.
        bps: Shape (B, N_BPS_POINTS, 3). It is important that we reuse the same BPS!
        anchor_assignment: How to assign anchors to BPS points. Must be one of: "random",
        "closest", "closest_and_farthest", "batched_fixed".
    """
    assert len(anchors.shape) == 3
    assert len(pointcloud.shape) == 3
    assert len(bps.shape) == 3
    bps_encoder = bps_torch(custom_basis=bps)
    # TODO: Investigate why parts of the pointcloud (i.e. in the wine glass) are ignored during
    # sampling, especially when reducing the dimensionality of the BPS representation.
    rescaled_obj_pointcloud = pointcloud * scalar
    rescaled_anchors = anchors * scalar
    object_bps: Dict[str, Any] = bps_encoder.encode(
        rescaled_obj_pointcloud,
        feature_type=["dists", "deltas"],
    )
    rescaled_ref_pts = bps + object_bps["deltas"]
    # Compute the distances between the BPS points and the MANO anchors:
    anchor_distances = torch.cdist(bps, rescaled_anchors)  # Shape: (BPS_LEN, N_ANCHORS)
    anchor_encodings = []
    if anchor_assignment == "random":
        # Randomly sample an anchor for each reference point:
        anchor_ids = torch.randint(
            0,
            rescaled_anchors.shape[1],
            (
                1,
                bps.shape[0],
            ),
            device=bps.device,
        )
        distances = torch.gather(anchor_distances, 2, anchor_ids.unsqueeze(-1)).squeeze(
            -1
        )
        anchor_encodings = [
            distances.unsqueeze(-1),
            torch.nn.functional.one_hot(
                anchor_ids, num_classes=rescaled_anchors.shape[1]
            ),
        ]
        # rescaled_anchors_repeats = (
        # torch.gather(rescaled_anchors, 1, anchor_ids.unsqueeze(-1).repeat(1, 1, 3))
        # .squeeze(-2)
        # .squeeze(0)
        # )
        # anchor_deltas = rescaled_anchors_repeats - bps
        # assert torch.allclose(anchor_deltas + bps, rescaled_anchors_repeats)
    elif anchor_assignment == "closest":
        anchor_ids = torch.argmin(anchor_distances, dim=2)
        distances = torch.gather(anchor_distances, 2, anchor_ids.unsqueeze(-1)).squeeze(
            -1
        )
        anchor_encodings = [
            distances.unsqueeze(-1),
            torch.nn.functional.one_hot(
                anchor_ids, num_classes=rescaled_anchors.shape[1]
            ),
        ]
        # rescaled_anchors_repeats = (
        # torch.gather(rescaled_anchors, 1, anchor_ids.unsqueeze(-1).repeat(1, 1, 3))
        # .squeeze(-2)
        # .squeeze(0)
        # )
        # anchor_deltas = rescaled_anchors_repeats - bps
        # assert torch.allclose(anchor_deltas + bps, rescaled_anchors_repeats)
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
                closest_anchor_ids, num_classes=rescaled_anchors.shape[1]
            ),
            farthest_distances.unsqueeze(-1),
            torch.nn.functional.one_hot(
                farthest_anchor_ids, num_classes=rescaled_anchors.shape[1]
            ),
        ]
        # anchor_deltas = torch.zeros_like(bps)  # Not implemented yet!
        # raise NotImplementedError(
        # "closest_and_farthest anchor assignment is not fully implemented yet!"
        # )
    elif anchor_assignment == "batched_fixed":
        # Assign all ordered 32 rescaled_anchors to a batch of BPS points and repeat for all available
        # batches. The number of batches is determined by the number of BPS points, and the latter
        # must be a multiple of 32.
        assert (
            bps.shape[1] % 32 == 0
        ), f"The number of BPS points ({bps.shape[1]}) must be a multiple of 32 for batched_fixed anchor assignment."
        anchor_ids = (
            torch.arange(
                0,
                rescaled_anchors.shape[1],
                device=bps.device,
            )
            .repeat((bps.shape[1] // 32,))
            .unsqueeze(0)
        )
        distances = torch.gather(anchor_distances, 2, anchor_ids.unsqueeze(-1)).squeeze(
            -1
        )
        # Here we won't need the anchor index since it's a fixed pattern!
        anchor_encodings = [
            distances.unsqueeze(-1),
        ]
        # rescaled_anchors_repeats = (
        # torch.gather(rescaled_anchors, 1, anchor_ids.unsqueeze(-1).repeat(1, 1, 3))
        # .squeeze(-2)
        # .squeeze(0)
        # )
        # anchor_deltas = rescaled_anchors_repeats - bps
        # assert torch.allclose(anchor_deltas + bps, rescaled_anchors_repeats)
    # Build the CHOIR representation:
    choir = torch.cat(
        [
            object_bps["dists"].unsqueeze(-1),
            *anchor_encodings,
        ],
        dim=-1,
    )
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
