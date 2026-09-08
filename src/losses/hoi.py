#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Hand-Object Interaction loss.
"""

from collections import deque
from functools import partial
from typing import Dict, Optional, Tuple

import torch

from utils.dataset import lower_tril_cholesky_to_covmat






class CHOIRFittingLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        # verts: torch.Tensor,
        anchors: torch.Tensor,
        choir: torch.Tensor,
        bps: torch.Tensor,
        anchor_indices: torch.Tensor,
        # bps_mean: torch.Tensor,
        # bps_scalar: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(bps.shape) == 2:
            assert bps.shape[0] % 32 == 0, "bps_dim must be a multiple of 32"
        elif len(bps.shape) == 3:
            assert bps.shape[1] % 32 == 0, "bps_dim must be a multiple of 32"
        """
        Args:
            anchors (torch.Tensor): (B, 32, 3) The predicted MANO anchors in the BPS coordinate system.
            choir (torch.Tensor): (B, P, 2) The predicted CHOIR field in the BPS coordinate system.
                                            P is the number of points in the BPS representation and
                                            the last dimension is composed of: (a) the nearest
                                            object BPS distance, (b) the MANO anchor BPS distance
                                            (not nearest but fixed anchors).
            bps (torch.Tensor): (B, P, 3) The BPS representation of the object point cloud. Since
                                            it is fixed, it would make sense to be (P, 3) but we
                                            rescale the BPS according to the scale of the
                                            hand-object pair associated with batch element n, so we
                                            duplicate the BPS for each batch element and apply the
                                            corresponding scalar.

        This loss should be computed as MSE between the computed distance
        from the predicted anchors to the reconstructed target point cloud and the encoded
        distances in the choir field.
        IMPORTANT: The anchors and the CHOIR field must be in the same coordinate frame and scale.
        When doing test-time optimization, we pass the rescaled BPS so that we can predict MANO in
        the camera frame and not in the original BPS frame.
        """
        # Step 1: Compute the distance from the predicted anchors to the reconstructed target point cloud
        # tgt_points = self.bps.decode(x_deltas=choir[:, :, 1:4])
        # tgt_points = denormalize(tgt_points, bps_mean, bps_scalar)
        if len(bps.shape) == 2:
            assert bps.shape[0] == choir.shape[1]
        elif len(bps.shape) == 3:
            assert bps.shape[1] == choir.shape[1]
        else:
            raise ValueError("CHOIRFittingLoss(): BPS must be (B, P, 3) or (P, 3)")
        assert (
            bps.shape[0] == anchors.shape[0]
        ), "optimize_pose_pca_from_choir(): BPS and anchors must have the same batch size"
        anchor_distances = torch.cdist(bps, anchors, p=2)
        anchor_ids = (
            anchor_indices.unsqueeze(0)
            .repeat((choir.shape[0], 1))
            .unsqueeze(-1)
            .type(torch.int64)
        )
        distances = torch.gather(anchor_distances, 2, anchor_ids).squeeze(-1)
        choir_loss = torch.nn.functional.mse_loss(distances, choir[:, :, -1])
        return choir_loss


class ContactsFittingLoss(torch.nn.Module):
    def __init__(
        self,
        init_verts: torch.Tensor,
        init_anchors: torch.tensor,
        update_knn_each_step: bool = True,
        use_median_filter: bool = False,
        median_filter_len: int = 15,
    ):
        """
        Args:
            update_knn_each_step (bool): If True, the KNN index buffer will be updated at each
                                         forward pass. If False, the buffer will be updated only once
                                         and the same KNN index buffer will be used for all forward
                                         passes.
            use_median_filter (bool): If True, the KNN index buffer will be used to compute the
                                     median of the K nearest neighbors at each step. This is useful
                                     to smooth the KNN index buffer and remove outliers.
            median_filter_len (int): The length of the median filter buffer.
        """
        super().__init__()
        self.knn = None
        self.update_knn_each_step = update_knn_each_step
        self.use_median_filter = use_median_filter
        assert not (use_median_filter and not update_knn_each_step), (
            "If you want to use the median filter, you must update the KNN index buffer at each "
            "step."
        )
        self.knn_index_buffer = deque(
            [], maxlen=median_filter_len
        )  # For the median filter
        with torch.no_grad():
            # Compute anchor-vertex indices as (32, V) where each element is the index of the
            # anchor which is closest to the corresponding vertex.
            dists = torch.cdist(init_verts, init_anchors)  # (V, 32)
            self.vertex_anchor_indices = torch.zeros(
                (32, init_verts.shape[0]), dtype=torch.int64
            )
            closest_anchor_indices = torch.argmin(dists, dim=-1)
            for i in range(32):
                # I want a vector of shape (V,) where each element is a boolean indicating whether
                # the vertex belongs to anchor i.
                anchor_indices = (closest_anchor_indices == i).squeeze(-1)
                self.vertex_anchor_indices[i] = anchor_indices
            # To booleans:
            self.vertex_anchor_indices = self.vertex_anchor_indices.bool()
        self.mvns = {}

    def _compute_contacts_loss(
        self,
        verts: torch.Tensor,
        anchor_verts: torch.Tensor,
        obj_pts: torch.Tensor,
        contact_gaussians: torch.Tensor,
        K: int = 5,  # TODO: Move to init
        weights_threshold: float = 0.01,  # TODO: Move to init
        scale_tril_norm_activation_threshold: float = 1e-3,  # TODO: Move to init
    ) -> torch.Tensor:
        # 1. Compute K object nearest neighbors for each MANO vertex
        if self.knn is None:
            neighbours = torch.cdist(verts, obj_pts)  # (V, O)
            knn, obj_pt_indices = torch.topk(
                neighbours, K, dim=-1, largest=False, sorted=False
            )  # (V, K)
            if self.use_median_filter:
                if len(self.knn_index_buffer) != self.knn_index_buffer.maxlen:
                    # Fill the buffer with the first indices
                    for _ in range(self.knn_index_buffer.maxlen):
                        self.knn_index_buffer.appendleft(obj_pt_indices)
                self.knn_index_buffer.appendleft(obj_pt_indices)
                obj_pt_indices = torch.median(
                    torch.stack(list(self.knn_index_buffer)), dim=0
                ).values
                knn = torch.gather(neighbours, -1, obj_pt_indices)
            if not self.update_knn_each_step:
                self.knn = knn
        else:  # Reuse
            knn = self.knn.detach()

        # 2. Compute the squared distance between each MANO vertex and its K nearest neighbors
        distances = knn

        # 3. Compute the distance weights as the PDF values of the contact Gaussians (using the
        # nearest Gaussian to each vertex).
        # =================================
        for i in range(32):
            # anchor_indices is a vector of shape (B, V, 1) where each element is the index of the
            # MANO anchor which is closest to the corresponding vertex in the batch.
            # We want to get the vertices which are associated with anchor i.
            this_anchor_indices = self.vertex_anchor_indices[i]
            neighbourhood_verts = verts[:, this_anchor_indices]
            if i not in self.mvns:
                skip_anchor = False
                # =====================================
                mean, cov = contact_gaussians[:, i, :3], contact_gaussians[:, i, 3:]
                scale_tril = None
                if cov.shape[-1] == 9:
                    cov = cov.view(-1, 3, 3)
                    if torch.norm(cov) < scale_tril_norm_activation_threshold**2:
                        skip_anchor = True
                elif cov.shape[-1] == 6:
                    # Cholesky-decomposed
                    scale_tril = (
                        lower_tril_cholesky_to_covmat(
                            cov[:, None, :], return_lower_tril=True
                        )
                        .squeeze(1)
                        .view(-1, 3, 3)
                    )
                    if torch.norm(scale_tril) < scale_tril_norm_activation_threshold:
                        skip_anchor = True
                else:
                    raise ValueError(
                        f"ContactsFittingLoss(): The covariance matrix must be of shape (B, N, 6) or (B, 9) but got {cov.shape}"
                    )
                if skip_anchor:
                    self.mvns[i] = None
                else:
                    # /!\ The mean is at the origin, so we shift it by the anchor verts:
                    if torch.any(torch.isnan(anchor_verts[:, i])):
                        raise ValueError("NaNs in anchor_verts")
                    mean = mean  # + anchor_verts[:, i]
                    bs, n = neighbourhood_verts.shape[:2]
                    mean = mean.unsqueeze(1).expand(bs, n, 3)
                    mvn = partial(torch.distributions.MultivariateNormal, loc=mean)
                    if scale_tril is None:
                        cov = cov.unsqueeze(1).expand(bs, n, 3, 3)
                        anchor_gaussian = mvn(covariance_matrix=cov)
                    else:
                        scale_tril = scale_tril.unsqueeze(1).expand(bs, n, 3, 3)
                        anchor_gaussian = mvn(scale_tril=scale_tril)
                    self.mvns[i] = anchor_gaussian

            if self.mvns[i] is None:
                distances[:, this_anchor_indices] *= 0
                continue
            weights = torch.exp(
                self.mvns[i].log_prob(neighbourhood_verts - anchor_verts[:, None, i])
            )
            # weights are (B, N) and we want to normalize them to [0, 1] for each batch element.
            min_, max_ = weights.min(dim=-1).values, weights.max(dim=-1).values
            max_ = max_ + 1e-6  # Avoid division by zero
            weights = (weights - min_.unsqueeze(-1)) / (max_ - min_).unsqueeze(-1)
            weights = torch.clamp(weights, 0, 1)  # Cause we may have +inf or -inf
            # Prune the very low weights:
            weights = torch.where(
                weights > weights_threshold, weights, torch.zeros_like(weights)
            )
            # 4. Compute the weighted sum of the distances as the loss.
            distances[:, this_anchor_indices] = (
                distances[:, this_anchor_indices] * weights[..., None]
            )
        return distances.mean()

    def _compute_penetration_loss(
        self,
        verts: torch.Tensor,
        obj_normals: torch.Tensor,
    ) -> torch.Tensor:
        # ========= Regularization =========
        # Penalize penetration of the hand into the object.
        # We can use the object normals to compute the dot product between the normals and the
        # direction of the vertices to the object points.
        # We first have to find the nearest object normal to each hand vertex:
        obj_vert_distances = torch.cdist(verts, obj_normals[..., :3])  # (V, O)
        nearest_normal_indices = torch.argmin(obj_vert_distances, dim=-1)  # (V,)
        nearest_normals = torch.take_along_dim(
            obj_normals, nearest_normal_indices.unsqueeze(-1), dim=1
        )
        nearest_normal_roots, nearest_normals = (
            nearest_normals[..., :3],
            nearest_normals[..., 3:],
        )
        # Shift the nearest normal roots 2mm inwards (2mm * normal direction, assuming unit normals) to avoid penalizing for the hand being
        # in direct contact with the object.
        # assert (
        #    torch.norm(nearest_normals) == 1
        # ), "The object normals must be unit vectors"
        # nearest_normals = torch.nn.functional.normalize(nearest_normals) # Trimesh already gives me normalized vectors
        nearest_normal_roots = nearest_normal_roots - 0.002 * nearest_normals
        dot_products = torch.einsum(
            "bvi,bvi->bv", (nearest_normals, verts - nearest_normal_roots)
        )
        # A negative dot product means the hand is penetrating the object. We allow a small
        # tolerance of 2mm as per the above shift.
        return (torch.nn.functional.relu(-dot_products)).mean()

    def forward(
        self,
        verts: torch.Tensor,
        anchor_verts: torch.Tensor,
        obj_pts: torch.Tensor,
        contact_gaussians: torch.Tensor,
        K: int = 5,  # TODO: Move to init
        weights_threshold: float = 0.01,  # TODO: Move to init
        scale_tril_norm_activation_threshold: float = 1e-3,  # TODO: Move to init
        obj_normals: Optional[torch.Tensor] = None,
        only_penetration_loss: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        TODO: Write this docstring.
        Args:
            obj_normals (torch.Tensor): (O, 6) The normals (0:3 for root, 3: for normal vector) of
                                    the object point cloud. If provided, the loss will include a
                                    regularization term to penalize penetration of the hand into
                                    the object.
        """
        contacts_loss, penetration_loss = torch.tensor(0.0), torch.tensor(0.0)
        if not only_penetration_loss:
            contacts_loss = self._compute_contacts_loss(
                verts,
                anchor_verts,
                obj_pts,
                contact_gaussians,
                K,
                weights_threshold,
                scale_tril_norm_activation_threshold,
            )

        if obj_normals is not None:
            penetration_loss = self._compute_penetration_loss(verts, obj_normals)
        # A simpler approach is to strongly penalize for a hand-to-object distance being below
        # 2mm, without any penalization above 1mm.
        # penetration_loss = (torch.nn.functional.relu(0.001 - distances) ** 2).mean()
        return contacts_loss, penetration_loss


