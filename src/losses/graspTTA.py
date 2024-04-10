#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2024 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
GraspTTA losses.
"""


import importlib

import chamfer_distance as chd
import numpy as np
import torch
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops.knn import knn_points
from pytorch3d.structures import Meshes

from utils import to_cuda_


def get_pseudo_cmap(nn_dists):
    """
    calculate pseudo contactmap: 0~3cm mapped into value 1~0
    :param nn_dists: object nn distance [B, N] or [N,] in meter**2
    :return: pseudo contactmap [B,N] or [N,] range in [0,1]
    """
    nn_dists = 100.0 * torch.sqrt(nn_dists)  # turn into center-meter
    cmap = 1.0 - 2 * (torch.sigmoid(nn_dists * 2) - 0.5)
    return cmap


def batched_index_select(input, index, dim=1):
    """
    :param input: [B, N1, *]
    :param dim: the dim to be selected
    :param index: [B, N2]
    :return: [B, N2, *] selected result
    """
    views = [input.size(0)] + [
        1 if i != dim else -1 for i in range(1, len(input.shape))
    ]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(input, dim=dim, index=index)


def get_interior(src_face_normal, src_xyz, trg_xyz, trg_NN_idx):
    """
    :param src_face_normal: [B, 778, 3], surface normal of every vert in the source mesh
    :param src_xyz: [B, 778, 3], source mesh vertices xyz
    :param trg_xyz: [B, 3000, 3], target mesh vertices xyz
    :param trg_NN_idx: [B, 3000], index of NN in source vertices from target vertices
    :return: interior [B, 3000], inter-penetrated trg vertices as 1, instead 0 (bool)
    """
    N1, N2 = src_xyz.size(1), trg_xyz.size(1)

    # get vector from trg xyz to NN in src, should be a [B, 3000, 3] vector
    NN_src_xyz = batched_index_select(src_xyz, trg_NN_idx)  # [B, 3000, 3]
    NN_vector = NN_src_xyz - trg_xyz  # [B, 3000, 3]

    # get surface normal of NN src xyz for every trg xyz, should be a [B, 3000, 3] vector
    NN_src_normal = batched_index_select(src_face_normal, trg_NN_idx)

    interior = (NN_vector * NN_src_normal).sum(
        dim=-1
    ) > 0  # interior as true, exterior as false
    return interior


def get_NN(src_xyz, trg_xyz, k=1):
    """
    :param src_xyz: [B, N1, 3]
    :param trg_xyz: [B, N2, 3]
    :return: nn_dists, nn_dix: all [B, 3000] tensor for NN distance and index in N2
    """
    B = src_xyz.size(0)
    src_lengths = torch.full(
        (src_xyz.shape[0],), src_xyz.shape[1], dtype=torch.int64, device=src_xyz.device
    )  # [B], N for each num
    trg_lengths = torch.full(
        (trg_xyz.shape[0],), trg_xyz.shape[1], dtype=torch.int64, device=trg_xyz.device
    )
    src_nn = knn_points(
        src_xyz, trg_xyz, lengths1=src_lengths, lengths2=trg_lengths, K=k
    )  # [dists, idx]
    nn_dists = src_nn.dists[..., 0]
    nn_idx = src_nn.idx[..., 0]
    return nn_dists, nn_idx


def contact_loss(obj_xyz, hand_xyz, cmap):
    """
    # prior cmap loss on gt cmap
    :param obj_xyz: [B, N1, 3]
    :param hand_xyz: [B, N2, 3]
    :param cmap: [B, N1] for contact map from NN dist thresholding
    :param hand_faces_index: [B, 1538, 3] hand index in [0,N2] for 3 vertices in a face
    :return:
    """

    # finger_vertices = [309, 317, 318, 319, 320, 322, 323, 324, 325,
    #                    326, 327, 328, 329, 332, 333, 337, 338, 339, 343, 347, 348, 349,
    #                    350, 351, 352, 353, 354, 355,  # 2nd finger
    #                    429, 433, 434, 435, 436, 437, 438, 439, 442, 443, 444, 455, 461, 462, 463, 465, 466,
    #                    467,  # 3rd
    #                    547, 548, 549, 550, 553, 566, 573, 578,  # 4th
    #                    657, 661, 662, 664, 665, 666, 667, 670, 671, 672, 677, 678, 683, 686, 687, 688, 689, 690, 691,
    #                    692, 693, 694, 695,  # 5th
    #                    736, 737, 738, 739, 740, 741, 743, 753, 754, 755, 756, 757, 759, 760, 761, 762, 763, 764, 766,
    #                    767, 768,  # 1st
    #                    73, 96, 98, 99, 772, 774, 775, 777]  # hand
    f1 = [ 697, 698, 699, 700, 712, 713, 714, 715, 737, 738, 739, 740, 741, 743, 744, 745, 746,
          748, 749, 750, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767,
          768, ]  # fmt: skip
    f2 = [ 46, 47, 48, 49, 164, 165, 166, 167, 194, 195, 223, 237, 238, 280, 281, 298, 301, 317,
          320, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 340, 341, 342, 343, 344, 345,
          346, 347, 348, 349, 350, 351, 352, 353, 354, 355, ]  # fmt: skip
    f3 = [ 356, 357, 358, 359, 375, 376, 386, 387, 396, 397, 402, 403, 413, 429, 433, 434, 435,
          436, 437, 438, 439, 440, 441, 442, 443, 444, 452, 453, 454, 455, 456, 459, 460, 461, 462,
          463, 464, 465, 466, 467, ]  # fmt: skip
    f4 = [ 468, 469, 470, 471, 484, 485, 486, 496, 497, 506, 507, 513, 514, 524, 545, 546, 547,
          548, 549, 550, 551, 552, 553, 555, 563, 564, 565, 566, 567, 570, 572, 573, 574, 575, 576,
          577, 578, ]  # fmt: skip
    f5 = [ 580, 581, 582, 583, 600, 601, 602, 614, 615, 624, 625, 630, 631, 641, 663, 664, 665,
          666, 667, 668, 670, 672, 680, 681, 682, 683, 684, 686, 687, 688, 689, 690, 691, 692, 693,
          694, 695, ]  # fmt: skip
    f0 = [73, 96, 98, 99, 772, 774, 775, 777]  # fmt: skip
    prior_idx = f1 + f2 + f3 + f4 + f5 + f0
    hand_xyz_prior = hand_xyz[
        :, prior_idx, :
    ]  # only using prior points for contact map

    B = obj_xyz.size(0)

    obj_CD, _ = get_NN(
        obj_xyz, hand_xyz_prior
    )  # [B, N1] NN distance from obj pc to hand pc

    # compute contact map loss
    n_points = torch.sum(cmap)
    cmap_loss = obj_CD[cmap].sum() / (B * n_points)

    return 3000.0 * cmap_loss


class GraspCVAELoss(torch.nn.Module):
    def __init__(
        self,
        a: float = 1.0,
        b: float = 0.1,
        c: float = 1000.0,
        d: float = 10.0,
        e: float = 10.0,
        hand_weights_path: str = "rhand_weights.npy",
    ):
        super().__init__()
        self.emd_module = None
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
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

    def CVAE_loss_mano(self, recon_x, x, mean, log_var, loss_tpye, mode="train"):
        """
        :param recon_x: reconstructed hand xyz [B,778,3]
        :param x: ground truth hand xyz [B,778,6]
        :param mean: [B,z]
        :param log_var: [B,z]
        :return:
        """
        if loss_tpye == "L2":
            recon_loss = torch.nn.functional.mse_loss(
                recon_x, x, reduction="none"
            ).sum() / x.size(0)
        elif loss_tpye == "CD":
            recon_loss, _ = chamfer_distance(
                recon_x, x, point_reduction="sum", batch_reduction="mean"
            )
        elif loss_tpye == "EMD":
            if self.emd_module is None:
                # I can't install it on MacOS so importing here to avoid errors
                self.emd_module = importlib.import_module(
                    "vendor.MSN-Point-Cloud-Completion.emd.emd_module"
                )
            emd = self.emd_module.emdModule()
            # recon_loss = earth_mover_distance(
            #    recon_x, x, transpose=False
            # ).sum() / x.size(0)
            dist, _ = emd(recon_x, x, 0.002, 1000)  # Defaults from the package
            recon_loss = dist.sum() / x.size(0)
        if mode != "train":
            return torch.tensor(0.0), recon_loss, torch.tensor(0.0)
        # KLD loss
        KLD = (
            -0.5
            * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
            / x.size(0)
            * 10.0
        )
        if mode == "train":
            return recon_loss + KLD, recon_loss, KLD

    def CMap_consistency_loss_soft(recon_hand_xyz, gt_hand_xyz, obj_xyz):
        recon_dists, _ = get_NN(obj_xyz, recon_hand_xyz)  # [B, N1]
        gt_dists, _ = get_NN(obj_xyz, gt_hand_xyz)  # [B, N1]
        consistency = torch.nn.functional.mse_loss(
            recon_dists, gt_dists, reduction="none"
        ).sum() / recon_dists.size(0)
        return consistency

    def CMap_consistency_loss(
        self, obj_xyz, recon_hand_xyz, gt_hand_xyz, recon_dists, gt_dists
    ):
        """
        :param recon_hand_xyz: [B, N2, 3]
        :param gt_hand_xyz: [B, N2, 3]
        :param obj_xyz: [B, N1, 3]
        :return:
        """
        # if not recon_dists or not gt_dists:
        #     recon_dists, _ = utils_loss.get_NN(obj_xyz, recon_hand_xyz)  # [B, N1]
        #     gt_dists, _ = utils_loss.get_NN(obj_xyz, gt_hand_xyz)  # [B, N1]
        recon_dists = torch.sqrt(recon_dists)
        gt_dists = torch.sqrt(gt_dists)
        # hard cmap
        recon_cmap = recon_dists < 0.005
        gt_cmap = gt_dists < 0.005
        gt_cpoint_num = gt_cmap.sum() + 0.0001
        consistency = (recon_cmap * gt_cmap).sum() / gt_cpoint_num
        # soft cmap
        # consistency2 = torch.nn.functional.mse_loss(recon_dists, gt_dists, reduction='none').sum() / recon_dists.size(0)
        return -5.0 * consistency  # + consistency2

    def inter_penetr_loss(self, hand_xyz, hand_face, obj_xyz, nn_dist, nn_idx):
        """
        get penetrate object xyz and the distance to its NN
        :param hand_xyz: [B, 778, 3]
        :param hand_face: [B, 1538, 3], hand faces vertex index in [0:778]
        :param obj_xyz: [B, 3000, 3]
        :return: inter penetration loss
        """
        B = hand_xyz.size(0)
        mesh = Meshes(verts=hand_xyz, faces=hand_face)
        hand_normal = mesh.verts_normals_packed().view(-1, 778, 3)

        # if not nn_dist:
        #     nn_dist, nn_idx = utils_loss.get_NN(obj_xyz, hand_xyz)
        interior = get_interior(hand_normal, hand_xyz, obj_xyz, nn_idx).type(
            torch.bool
        )  # True for interior
        penetr_dist = nn_dist[interior].sum() / B  # batch reduction
        return 100.0 * penetr_dist

    def forward(
        self,
        recon_x,
        x,
        mu,
        logvar,
        recon_xyz,
        hand_xyz,
        hand_faces,
        obj_pts,
        epoch: int,
        training_mode=True,
    ):
        # L2 MANO params loss
        param_loss = torch.nn.functional.mse_loss(
            recon_x, x, reduction="none"
        ).sum() / recon_x.size(0)

        cvae_loss, recon_loss, kld_loss = self.CVAE_loss_mano(
            recon_xyz, hand_xyz, mu, logvar, "CD", "train" if training_mode else "eval"
        )

        if training_mode:
            # obj xyz NN dist and idx
            obj_nn_dist_gt, obj_nn_idx_gt = get_NN(obj_pts, hand_xyz)
            obj_nn_dist_recon, obj_nn_idx_recon = get_NN(obj_pts, recon_xyz)

            cmap_loss = contact_loss(obj_pts, recon_xyz, obj_nn_dist_recon < 0.01**2)
            # cmap consistency loss
            consistency_loss = self.CMap_consistency_loss(
                obj_pts,
                recon_xyz,
                hand_xyz,
                obj_nn_dist_recon,
                obj_nn_dist_gt,
            )
            # inter penetration loss
            penetr_loss = self.inter_penetr_loss(
                recon_xyz,
                hand_faces,
                obj_pts,
                obj_nn_dist_recon,
                obj_nn_idx_recon,
            )

            loss = (
                self.a * cvae_loss
                + self.b * param_loss
                + self.d * penetr_loss
                + self.e * consistency_loss
            )
            if epoch >= 5:
                loss += self.c * cmap_loss
            return loss, {
                "param_loss": param_loss,
                "recon_loss": recon_loss,
                "KLD_loss": kld_loss,
                "cam_loss": cmap_loss,
                "penetr_loss": penetr_loss,
                "consistency_loss": consistency_loss,
            }
        else:
            loss = param_loss + recon_loss
            return loss, {
                "param_loss": param_loss,
                "recon_loss": recon_loss,
            }


def TTT_loss(hand_xyz, hand_face, obj_xyz, cmap_affordance, cmap_pointnet):
    """
    :param hand_xyz:
    :param hand_face:
    :param obj_xyz:
    :param cmap_affordance: contact map calculated from predicted hand mesh
    :param cmap_pointnet: target contact map predicted from ContactNet
    :return:
    """
    B = hand_xyz.size(0)

    # inter-penetration loss
    mesh = Meshes(verts=to_cuda_(hand_xyz), faces=to_cuda_(hand_face))
    hand_normal = mesh.verts_normals_packed().view(-1, 778, 3)
    nn_dist, nn_idx = get_NN(obj_xyz, hand_xyz)
    interior = get_interior(hand_normal, hand_xyz, obj_xyz, nn_idx).type(torch.bool)
    penetr_dist = 120 * nn_dist[interior].sum() / B  # batch reduction

    # cmap consistency loss
    consistency_loss = (
        0.0001
        * torch.nn.functional.mse_loss(
            cmap_affordance, cmap_pointnet, reduction="none"
        ).sum()
        / B
    )

    # hand-centric loss
    cl = 2.5 * contact_loss(obj_xyz, hand_xyz, cmap=nn_dist < 0.01**2)
    return penetr_dist, consistency_loss, cl
