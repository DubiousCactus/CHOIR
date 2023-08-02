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
from pytorch3d.structures import Pointclouds
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
        )
        .unsqueeze(0)
        .cuda()
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


def compute_choir(
    pointcloud: torch.Tensor,
    anchors: torch.Tensor,
    scalar: float,
    bps: torch.Tensor,
    remap_bps_distances: bool,
    exponential_map_w: float,
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
    rescaled_ref_pts = bps + object_bps["deltas"]  # A subset of rescaled_obj_pointcloud
    # Compute the distances between the BPS points and the MANO anchors:
    anchor_distances = torch.cdist(bps, rescaled_anchors)  # Shape: (BPS_LEN, N_ANCHORS)
    # Assign all ordered 32 rescaled_anchors to a batch of BPS points and repeat for all available
    # batches. The number of batches is determined by the number of BPS points, and the latter
    # must be a multiple of 32.
    assert (
        bps.shape[1] % 32 == 0
    ), f"The number of BPS points ({bps.shape[1]}) must be a multiple of 32."
    anchor_ids = (
        torch.arange(
            0,
            rescaled_anchors.shape[1],
            device=bps.device,
        )
        .repeat((bps.shape[1] // 32,))
        .unsqueeze(0)
        .repeat((B, 1))
    )
    distances = torch.gather(anchor_distances, 2, anchor_ids.unsqueeze(-1)).squeeze(-1)
    choir = torch.cat(
        [
            object_bps["dists"]
            .unsqueeze(-1)
            .repeat(
                (B, 1, 1)
            ),  # O(len(BPS)) because the anchor distances don't involve KNN.
            distances.unsqueeze(-1),
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
