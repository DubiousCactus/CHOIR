#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Hand-Object Interaction loss.
"""

from typing import Dict, Tuple

import torch

from model.affine_mano import AffineMANO


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
    def __init__(self):
        super().__init__()

    def forward(
        self,
        verts: torch.Tensor,
        anchor_verts: torch.Tensor,
        obj_pts: torch.Tensor,
        contact_gaussians: torch.Tensor,
        K: int = 5,
        weights_threshold: float = 0.01,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        TODO: Write this docstring.
        """
        # 1. Compute K object nearest neighbors for each MANO vertex
        knn = torch.cdist(verts, obj_pts)  # (V, O)
        knn = torch.topk(knn, K, dim=-1, largest=False, sorted=False).values  # (V, K)

        # 2. Compute the squared distance between each MANO vertex and its K nearest neighbors
        distances = knn

        # 3. Compute the distance weights as the PDF values of the contact Gaussians (using the
        # nearest Gaussian to each vertex).
        anchor_distances = torch.cdist(verts, anchor_verts)  # (V, A)
        anchor_distances, anchor_indices = torch.topk(
            anchor_distances, 1, dim=-1, largest=False, sorted=False
        )
        for i in range(32):
            # Find vertices associated with anchor i
            # print(
            # f"verts: {verts.shape}, indices: {anchor_indices.shape}, "
            # + f"this_anchor_indices: {(anchor_indices==i).shape}, "
            # + f"contact_gaussians: {contact_gaussians.shape}"
            # )
            this_anchor_indices = (anchor_indices == i).squeeze(-1)
            neighbourhood_verts = verts[this_anchor_indices]
            # print(f"neighbourhood_verts: {neighbourhood_verts.shape}")
            # if torch.allclose(
            # contact_gaussians[i], torch.zeros_like(contact_gaussians[i]), rtol=1e-9
            # ):
            if torch.all(contact_gaussians[i] == 0):
                # print(f"anchor {i} has not enough contacts")
                distances[this_anchor_indices] *= 0
                continue
            mean, cov = contact_gaussians[i, :3], contact_gaussians[i, 3:].reshape(3, 3)
            # /!\ The mean is at the origin, so we shift it by the anchor verts:
            mean = mean + anchor_verts[:, i]
            anchor_gaussian = torch.distributions.MultivariateNormal(mean, cov)
            weights = torch.exp(anchor_gaussian.log_prob(neighbourhood_verts))
            print(f"Max contact values for Gaussian {i}/32: {weights.max()}")
            # print(
            # f"weights: {weights.shape}. neighbour verts: {neighbourhood_verts.shape}. "
            # + f"distances: {distances[this_anchor_indices].shape}"
            # )
            # weights = torch.exp(torch.clamp_min(weights, -200))
            print(f"weights range: {weights.min()} - {weights.max()}")
            weights = (weights / weights.max()) if weights.max() > 1.0 else weights
            print(f"weights new range: {weights.min()} - {weights.max()}")
            # Prune the very low weights:
            weights = torch.where(
                weights > weights_threshold, weights, torch.zeros_like(weights)
            )
            print(f"Pruned weights range: {weights.min()} - {weights.max()}")
            print(
                f"distances range: {distances[this_anchor_indices].min()} - {distances[this_anchor_indices].max()}"
            )
            distances[this_anchor_indices] = (
                distances[this_anchor_indices] * weights[..., None].detach()
            )
            print(
                f"distances new range: {distances[this_anchor_indices].min()} - {distances[this_anchor_indices].max()}"
            )
        # 4. Compute the weighted sum of the distances as the loss.
        print(f"distances: {distances.shape}")
        print(f"distances range: {distances.min()} - {distances.max()}")
        print(
            f"distances above 5mm: {torch.where(distances > 0.005, 1.0, 0.0).sum().item()}"
        )
        return (distances**2).mean()
