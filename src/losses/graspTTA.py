#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2024 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
GraspTTA losses.
"""


import chamfer_distance as chd
import numpy as np
import torch
from pytorch3d.structures import Meshes


class GraspCVAELoss(torch.nn.Module):
    def __init__(
        self,
        a: float = 1.0,
        b: float = 0.1,
        c: float = 10.0,
        hand_weights_path: str = "rhand_weights.npy",
    ):
        super().__init__()
        self.a = a
        self.b = b
        self.c = c
        self.v_weights = torch.from_numpy(np.load(hand_weights_path)).to(torch.float32)

    def point2point_signed(
        self,
        x,
        y,
        x_normals=None,
        y_normals=None,
    ):
        """
        signed distance between two pointclouds
        Args:
            x: FloatTensor of shape (N, P1, D) representing a batch of point clouds
                with P1 points in each batch element, batch size N and feature
                dimension D.
            y: FloatTensor of shape (N, P2, D) representing a batch of point clouds
                with P2 points in each batch element, batch size N and feature
                dimension D.
            x_normals: Optional FloatTensor of shape (N, P1, D).
            y_normals: Optional FloatTensor of shape (N, P2, D).
        Returns:
            - y2x_signed: Torch.Tensor
                the sign distance from y to x
            - y2x_signed: Torch.Tensor
                the sign distance from y to x
            - yidx_near: Torch.tensor
                the indices of x vertices closest to y
        """

        N, P1, D = x.shape
        P2 = y.shape[1]

        if y.shape[0] != N or y.shape[2] != D:
            raise ValueError("y does not have the correct shape.")

        ch_dist = chd.ChamferDistance()

        x_near, y_near, xidx_near, yidx_near = ch_dist(x, y)

        xidx_near_expanded = xidx_near.view(N, P1, 1).expand(N, P1, D).to(torch.long)
        x_near = y.gather(1, xidx_near_expanded)

        yidx_near_expanded = yidx_near.view(N, P2, 1).expand(N, P2, D).to(torch.long)
        y_near = x.gather(1, yidx_near_expanded)

        x2y = x - x_near
        y2x = y - y_near

        if x_normals is not None:
            y_nn = x_normals.gather(1, yidx_near_expanded)
            in_out = (
                torch.bmm(
                    y_nn.contiguous().view(-1, 1, 3), y2x.contiguous().view(-1, 3, 1)
                )
                .contiguous()
                .view(N, -1)
                .sign()
            )
            y2x_signed = y2x.norm(dim=2) * in_out

        else:
            y2x_signed = y2x.norm(dim=2)

        if y_normals is not None:
            x_nn = y_normals.gather(1, xidx_near_expanded)
            in_out_x = (
                torch.bmm(x_nn.view(-1, 1, 3), x2y.view(-1, 3, 1)).view(N, -1).sign()
            )
            x2y_signed = x2y.norm(dim=2) * in_out_x
        else:
            x2y_signed = x2y.norm(dim=2)

        return y2x_signed, x2y_signed, yidx_near

    def loss_cnet(self, verts_rhand, verts_rhand_gt, faces, verts_obj, v_weights):
        device = verts_rhand.device
        dtype = verts_rhand.dtype

        B, NPoints_obj, _ = verts_obj.size()
        kl_coef = 0.005  # 5e-3
        v_weights2 = torch.pow(v_weights, 1.0 / 2.5)

        rh_mesh = (
            Meshes(verts=verts_rhand, faces=faces)
            .to(device)
            .verts_normals_packed()
            .view(-1, 778, 3)
        )
        rh_mesh_gt = (
            Meshes(verts=verts_rhand_gt, faces=faces)
            .to(device)
            .verts_normals_packed()
            .view(-1, 778, 3)
        )

        o2h_signed, h2o, _ = self.point2point_signed(verts_rhand, verts_obj, rh_mesh)
        o2h_signed_gt, h2o_gt, o2h_idx = self.point2point_signed(
            verts_rhand_gt, verts_obj, rh_mesh_gt
        )

        # addaptive weight for penetration and contact verts
        w_dist = (o2h_signed_gt < 0.01) * (o2h_signed_gt > -0.005)
        w_dist_neg = o2h_signed < 0.0
        w = torch.ones([B, NPoints_obj]).to(device)
        w[~w_dist] = 0.1  # less weight for far away vertices
        w[w_dist_neg] = 1.5  # more weight for penetration
        ######### dist loss
        loss_dist_h = (
            35
            * (1.0 - kl_coef)
            * torch.mean(
                torch.einsum(
                    "ij,j->ij", torch.abs(h2o.abs() - h2o_gt.abs()), v_weights2
                )
            )
        )
        loss_dist_o = (
            30
            * (1.0 - kl_coef)
            * torch.mean(
                torch.einsum("ij,ij->ij", torch.abs(o2h_signed - o2h_signed_gt), w)
            )
        )
        return loss_dist_h.sum() + loss_dist_o.sum()

    def forward(self, recon_x, x, mu, logvar, recon_xyz, hand_xyz, hand_faces, obj_pts):
        # L2 MANO params loss
        param_loss = torch.nn.functional.mse_loss(
            recon_x, x, reduction="none"
        ).sum() / recon_x.size(0)

        # L2 mesh reconstruction loss
        recon_loss = torch.nn.functional.mse_loss(
            recon_xyz, hand_xyz, reduction="none"
        ).sum() / hand_xyz.size(0)

        if mu is None or logvar is None:
            KLD = torch.tensor(0.0).to(recon_xyz.device)
        else:
            KLD = (
                -0.5
                * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                / hand_xyz.size(0)
            )
        cvae_loss = recon_loss + KLD
        ho_loss = self.loss_cnet(
            recon_xyz,
            hand_xyz,
            hand_faces,
            obj_pts,
            self.v_weights.to(recon_xyz.device),
        )

        loss = self.a * cvae_loss + self.b * param_loss + self.c * ho_loss
        return loss, {
            "param_loss": param_loss,
            "ho_loss": ho_loss,
            "recon_loss": recon_loss,
            "KLD_loss": KLD,
        }
