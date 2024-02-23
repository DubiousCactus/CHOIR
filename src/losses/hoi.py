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

from model.affine_mano import AffineMANO
from utils.dataset import lower_tril_cholesky_to_covmat


def kl_normal(qm, qv, pm, pv):
    # From https://github.com/davrempe/humor/blob/main/humor/losses/humor_loss.py#L359
    """
    Computes the elem-wise KL divergence between two normal distributions KL(q || p) and
    sum over the last dimension
    ​
    Args:
        qm: tensor: (batch, dim): q mean
        qv: tensor: (batch, dim): q variance
        pm: tensor: (batch, dim): p mean
        pv: tensor: (batch, dim): p variance
    Return:
        kl: tensor: (batch,): kl between each sample
    """
    element_wise = 0.5 * (
        torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1
    )
    kl = element_wise.sum(-1)
    return kl


class CHOIRLoss(torch.nn.Module):
    def __init__(
        self,
        bps: torch.Tensor,
        predict_anchor_orientation: bool,
        predict_anchor_position: bool,
        predict_mano: bool,
        orientation_w: float,
        distance_w: float,
        assignment_w: float,
        mano_pose_w: float,
        mano_global_pose_w: float,
        mano_shape_w: float,
        mano_agreement_w: float,
        mano_anchors_w: float,
        kl_w: float,
        multi_view: bool,
        temporal: bool,
        remap_bps_distances: bool,
        exponential_map_w: float,
        use_kl_scheduler: bool,
    ) -> None:
        super().__init__()
        self._mse = torch.nn.MSELoss(reduction="mean")
        self._cosine_embedding = torch.nn.CosineEmbeddingLoss()
        self._cross_entropy = torch.nn.CrossEntropyLoss()
        self._predict_anchor_orientation = predict_anchor_orientation
        self._predict_anchor_position = predict_anchor_position
        self._predict_mano = predict_mano
        self._orientation_w = orientation_w
        self._distance_w = distance_w
        self._assignment_w = assignment_w
        self._mano_pose_w = mano_pose_w
        self._mano_global_pose_w = mano_global_pose_w
        self._mano_shape_w = mano_shape_w
        self._mano_agreement_w = mano_agreement_w
        self._mano_anchors_w = mano_anchors_w
        self._kl_w = kl_w
        self._affine_mano = AffineMANO()
        self._hoi_loss = CHOIRFittingLoss() if predict_mano else None
        self._multi_view = multi_view
        self.register_buffer("bps", bps)
        self._remap_bps_distances = remap_bps_distances
        self._exponential_map_w = exponential_map_w
        self._kl_decay = 1.0 if not use_kl_scheduler else 1.2
        self._decayed = False
        self._temporal = temporal

    def forward(
        self,
        samples: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor],
        y_hat,
        epoch: int,
        rescale: bool = False,
    ) -> torch.Tensor:
        scalar = samples["scalar"]
        if len(scalar.shape) == 2:
            scalar = scalar.mean(dim=1)
        (
            choir_gt,
            scalar_gt,
            joints_gt,
            anchors_gt,
            pose_gt,
            beta_gt,
            rot_gt,
            trans_gt,
        ) = (
            labels["choir"],
            labels["scalar"],
            labels["joints"],
            labels["anchors"],
            labels["theta"],
            labels["beta"],
            labels["rot"],
            labels["trans"],
        )
        if self._multi_view and not self._temporal:
            for n in range(choir_gt.shape[1] - 1):
                assert torch.allclose(choir_gt[:, n], choir_gt[:, n + 1])
            # Since all noisy views are of the same grasp, we can just take the first view's ground
            # truth (all views have the same ground truth).
            choir_gt = choir_gt[:, 0]
            if len(scalar_gt.shape) == 2:
                scalar_gt = scalar_gt[
                    :, 0
                ]  # TODO: Make this consistant in data creation!
            pose_gt = pose_gt[:, 0]
            beta_gt = beta_gt[:, 0]
            rot_gt = rot_gt[:, 0]
            trans_gt = trans_gt[:, 0]
            anchors_gt = anchors_gt[:, 0]

        elif self._temporal:
            # Let's try to just predict the last frame's CHOIR field:
            choir_gt = choir_gt[:, -1]
            if len(scalar_gt.shape) == 2:
                scalar_gt = scalar_gt[:, -1]
            pose_gt = pose_gt[:, -1]
            beta_gt = beta_gt[:, -1]
            rot_gt = rot_gt[:, -1]
            trans_gt = trans_gt[:, -1]
            anchors_gt = anchors_gt[:, -1]

        choir_pred, orientations_pred = y_hat["choir"], y_hat["orientations"]

        losses = {
            "distances": self._distance_w
            * self._mse(choir_gt[..., 1], choir_pred[..., 1])
        }
        anchor_positions_pred = None
        if y_hat["posterior"] is not None and y_hat["prior"] is not None:
            kl_w = min(1e-5, self._kl_w * self._kl_decay ** (epoch // 10))
            if y_hat["posterior"][0] is not None:
                losses["kl_div"] = (
                    kl_normal(
                        y_hat["posterior"][0],
                        y_hat["posterior"][1],
                        y_hat["prior"][0],
                        y_hat["prior"][1],
                    ).mean()
                    * kl_w
                )
                # Strangely, I get much faster/better convergence with kl_normal() than with
                # Pytorch's kl_divergence() function, EVEN THOUGH the following assertion passes
                # (there is a negligible difference though, within atol or rtol). Why is that???
                # Like for real, the loss curves look SO DIFFERENT. They seem to converge to a
                # close point but kl_normal() might give me better results in the end (I need
                # thorough testing).
                # Investigate *after* the deadline ;)
                # Well for one the implementation *actually* differs from the official one. But
                # then who made a mistake? And maybe there's no mistake but maybe it's from p to q
                # and not q to p that we measure the KLdiv... Although in HUMOR paper it's clearly
                # q to p (posterior to prior).
                # official = (
                # kl_divergence(
                # torch.distributions.Normal(
                # y_hat["posterior"][0], y_hat["posterior"][1]
                # ),
                # torch.distributions.Normal(
                # y_hat["prior"][0], y_hat["prior"][1]
                # ),
                # ).mean()
                # * kl_w
                # )
                # # unofficial = kl_normal( y_hat["posterior"][0], y_hat["posterior"][1], y_hat["prior"][0], y_hat["prior"][1],).mean() * kl_w
                # # assert torch.allclose(official, unofficial)
        if self._predict_anchor_orientation or self._predict_anchor_position:
            raise NotImplementedError
            B, P, D = orientations_pred.shape
            losses["orientations"] = self._orientation_w * self._cosine_embedding(
                orientations_pred.reshape(B * P, D),
                orientations_gt.reshape(B * P, D),
                torch.ones(B * P, device=orientations_pred.device),
            )
            if self._predict_anchor_position:
                target_anchor_positions = orientations_gt * choir_gt[:, :, 4].unsqueeze(
                    -1
                ).repeat(1, 1, 3)
                anchor_positions_pred = orientations_pred * choir_pred[
                    :, :, 4
                ].unsqueeze(-1).repeat(1, 1, 3)
                losses["positions"] = self._distance_w * self._mse(
                    target_anchor_positions, anchor_positions_pred
                )
        if self._predict_mano:
            mano = y_hat["mano"]
            pose, shape, rot, trans = (
                mano[:, :18],
                mano[:, 18 : 18 + 10],
                mano[:, 18 + 10 : 18 + 10 + 6],
                mano[:, 18 + 10 + 6 :],
            )
            # Penalize high shape values to avoid exploding shapes and unnatural hands:
            shape_reg = torch.norm(shape, dim=-1).mean()  # ** 2
            # The MANO parameters and resulting anchor vertices are in the BPS coordinate system
            # and scale, such that the agreement loss is valid to the predicted CHOIR field.
            # Alternatively, we can rescale the CHOIR field to the original MANO coordinate system and
            # this way we can supervise the MANO parameters in the original coordinate system.
            verts, _ = self._affine_mano(pose, shape, rot, trans)
            anchors = self._affine_mano.get_anchors(verts)

            if self._remap_bps_distances:
                choir_pred = -torch.log(choir_pred) / self._exponential_map_w
            choir_pred = choir_pred / scalar[:, None, None]
            anchor_agreement_loss = self._hoi_loss(
                anchors,
                choir_pred,
                self.bps.unsqueeze(0).repeat(choir_pred.shape[0], 1, 1)
                / scalar[:, None, None],
            )
            losses["mano_pose"] = self._mano_pose_w * self._mse(pose, pose_gt)
            losses["mano_shape"] = self._mano_shape_w * (
                shape_reg + self._mse(shape, beta_gt)
            )
            losses["mano_global_pose"] = self._mano_global_pose_w * (
                self._mse(rot, rot_gt) + self._mse(trans, trans_gt)
            )
            losses["mano_anchors"] = self._mano_anchors_w * self._mse(
                anchors, anchors_gt
            )
            losses["mano_anchor_agreement"] = (
                self._mano_agreement_w * anchor_agreement_loss
            )
            # TODO: Penalize MANO penetration into the object pointcloud (recoverable through the
            # input CHOIR field which must include the delta vectors, or we can pass the delta
            # vectors to this loss separately).
        return losses


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

    def forward(
        self,
        verts: torch.Tensor,
        anchor_verts: torch.Tensor,
        obj_pts: torch.Tensor,
        contact_gaussians: torch.Tensor,
        K: int = 5,  # TODO: Move to init
        weights_threshold: float = 0.01,  # TODO: Move to init
        gaussian_activation_threshold: float = 1e-9,  # TODO: Move to init
        obj_normals: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        TODO: Write this docstring.
        Args:
            obj_normals (torch.Tensor): (O, 6) The normals (0:3 for root, 3: for normal vector) of
                                    the object point cloud. If provided, the loss will include a
                                    regularization term to penalize penetration of the hand into
                                    the object.
        """
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
        else:
            knn = self.knn

        # 2. Compute the squared distance between each MANO vertex and its K nearest neighbors
        distances = knn

        # 3. Compute the distance weights as the PDF values of the contact Gaussians (using the
        # nearest Gaussian to each vertex).
        penetration_loss = torch.tensor(0.0)
        # TODO: Vectorize this loop with masking
        # =================================
        for i in range(32):
            # anchor_indices is a vector of shape (B, V, 1) where each element is the index of the
            # MANO anchor which is closest to the corresponding vertex in the batch.
            # We want to get the vertices which are associated with anchor i.
            this_anchor_indices = self.vertex_anchor_indices[i]
            neighbourhood_verts = verts[:, this_anchor_indices]
            # =====================================
            if torch.allclose(
                contact_gaussians[:, i],
                torch.zeros_like(contact_gaussians[:, i]),
                atol=gaussian_activation_threshold,
            ):
                # print(f"anchor {i} has not enough contacts: {contact_gaussians[:, i]}")
                distances[:, this_anchor_indices] *= 0
                continue
            mean, cov = contact_gaussians[:, i, :3], contact_gaussians[:, i, 3:]
            scale_tril = None
            if cov.shape[-1] == 9:
                cov = cov.view(-1, 3, 3)
            elif cov.shape[-1] == 6:
                # Cholesky-decomposed
                scale_tril = (
                    lower_tril_cholesky_to_covmat(
                        cov[:, None, :], return_lower_tril=True
                    )
                    .squeeze(1)
                    .view(-1, 3, 3)
                )
                diag_flat_indices = [0, 4, 8]
                # TODO: Actually learn log-diagonal of the Cholesky factors?
                scale_tril = scale_tril.flatten(1)
                scale_tril[:, diag_flat_indices] = torch.relu(
                    scale_tril[:, diag_flat_indices]
                ) + torch.max(
                    scale_tril[:, diag_flat_indices],
                    torch.tensor(1e-9, device=scale_tril.device),
                )
                scale_tril = scale_tril.view(-1, 3, 3)
            else:
                raise ValueError(
                    f"ContactsFittingLoss(): The covariance matrix must be of shape (B, N, 6) or (B, 9) but got {cov.shape}"
                )
            # /!\ The mean is at the origin, so we shift it by the anchor verts:
            mean = mean + anchor_verts[:, i]
            bs, n = neighbourhood_verts.shape[:2]
            mean = mean.unsqueeze(1).expand(bs, n, 3)
            mvn = partial(torch.distributions.MultivariateNormal, loc=mean)
            if scale_tril is None:
                cov = cov.unsqueeze(1).expand(bs, n, 3, 3)
                anchor_gaussian = mvn(covariance_matrix=cov)
            else:
                scale_tril = scale_tril.unsqueeze(1).expand(bs, n, 3, 3)
                anchor_gaussian = mvn(scale_tril=scale_tril)
            weights = torch.exp(anchor_gaussian.log_prob(neighbourhood_verts))
            # print(f"Max contact values for Gaussian {i}/32: {weights.max()}")
            # weights are (B, N) and we want to normalize them to [0, 1] for each batch element.
            weights = (weights - weights.min(dim=-1).values.unsqueeze(-1)) / (
                weights.max(dim=-1).values - weights.min(dim=-1).values
            ).unsqueeze(-1)
            # print(f"weights new range: {weights.min(dim=-1).values} - {weights.max(dim=-1).values}")
            # Prune the very low weights:
            weights = torch.where(
                weights > weights_threshold, weights, torch.zeros_like(weights)
            )
            # 4. Compute the weighted sum of the distances as the loss.
            distances[:, this_anchor_indices] = (
                distances[:, this_anchor_indices] * weights[..., None]
            )

        # ========= Regularization =========
        # 5. Penalize penetration of the hand into the object.
        # We can use the object normals to compute the dot product between the normals and the
        # direction of the vertices to the object points.
        # penetration_loss = torch.tensor(0.0)
        if obj_normals is not None:
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
            # Shift the nearest normal roots 2mm inwards to avoid penalizing for the hand being
            # in direct contact with the object.
            nearest_normal_roots = nearest_normal_roots - 0.002 * nearest_normals
            dot_products = torch.einsum(
                "bvi,bvi->bv", (nearest_normals, verts - nearest_normal_roots)
            )
            # A negative dot product means the hand is penetrating the object. We allow a small
            # tolerance of 2mm as per the above shift.
            penetration_loss = (torch.nn.functional.relu(-dot_products)).mean()
        # A simpler approach is to strongly penalize for a hand-to-object distance being below
        # 2mm, without any penalization above 1mm.
        # penetration_loss = (torch.nn.functional.relu(0.001 - distances) ** 2).mean()
        return distances.mean(), penetration_loss
