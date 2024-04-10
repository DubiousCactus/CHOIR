#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2024 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
GraspTTA wrapper aroung GraspCVAE and ContactNet.
"""

from typing import Optional

import torch
from tqdm import tqdm

import conf.project as project_conf
from model.affine_mano import AffineMANO
from model.graspTTA.affordanceNet_obman_mano_vertex import affordanceNet
from model.graspTTA.ContactNet import pointnet_reg
from src.losses.graspTTA import TTT_loss, get_NN, get_pseudo_cmap
from utils import colorize


class GraspTTA(torch.nn.Module):
    def __init__(
        self,
        mano_params_dim: int,
        n_pts: int,
        tto_steps: int,
        graspCVAE_model_pth: Optional[str] = None,
        contactnet_model_pth: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        self.graspcvae = affordanceNet(mano_params_dim=mano_params_dim, **kwargs)
        self.contactnet = pointnet_reg(n_pts=n_pts, **kwargs)
        print(
            colorize(
                "Creating GraspTTA for ContactPose only!",
                project_conf.ANSI_COLORS["red"],
            )
        )
        self.affine_mano = AffineMANO(for_contactpose=True)  # TODO: OakInk?
        self.graspcvae.eval()
        self.contactnet.eval()
        self.tto_steps = tto_steps
        self._graspCVAE_model_pth = graspCVAE_model_pth
        self._contactnet_model_pth = contactnet_model_pth
        self.single_modality = "object"

    def forward(self, obj_pts):
        """
        :param obj_pc: [B, N1, 3+n]
        :return: reconstructed hand vertex
        """
        if (
            self._graspCVAE_model_pth is not None
            and self._contactnet_model_pth is not None
        ):
            print(
                colorize(
                    "[*] Manually loading GraspTTA models...",
                    project_conf.ANSI_COLORS["green"],
                )
            )
            print(
                colorize(
                    f"-> Loading GraspCVAE from {self._graspCVAE_model_pth}",
                    project_conf.ANSI_COLORS["green"],
                )
            )
            device = next(self.graspcvae.parameters()).device
            self.graspcvae.load_state_dict(
                torch.load(self._graspCVAE_model_pth, map_location=device)["model_ckpt"]
            )
            print(
                colorize(
                    f"-> Loading ContactNet from {self._contactnet_model_pth}",
                    project_conf.ANSI_COLORS["green"],
                )
            )
            self.contactnet.load_state_dict(
                torch.load(self._contactnet_model_pth, map_location=device)[
                    "model_ckpt"
                ]
            )
            self.graspcvae.eval()
            self.contactnet.eval()
            self._graspCVAE_model_pth = None
            self._contactnet_model_pth = None
            # Freeze ContactNet:
            for param in self.contactnet.parameters():
                param.requires_grad = False
            # Enable grads for GraspCVAE's decoder:
            for param in self.graspcvae.parameters():
                param.requires_grad = False
            for param in self.graspcvae.cvae.decoder.parameters():
                param.requires_grad = True

        B = obj_pts.size(0)
        with torch.no_grad():
            z = torch.randn(
                [obj_pts.size(0), self.graspcvae.cvae.latent_size],
                device=obj_pts.device,
            )
            obj_features, _, _ = self.graspcvae.obj_encoder(obj_pts.permute(0, 2, 1))
        # recon_param.requires_grad = True
        # optimizer = torch.optim.SGD([recon_param], lr=0.00000625, momentum=0.8)
        optimizer = torch.optim.Adam(self.graspcvae.cvae.decoder.parameters(), lr=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
        # TODO: The original implementation applies 1e6 random rotations to the object to obtain
        # variable grasps. Here we'll just go with canonical object pose, as our method is doing.
        # If we need to sample more grasps, we can just call this function multiple times. But in
        # effect, the GraspCVAE samples z from a random normal, so why would random rotation be
        # needed?

        pbar = tqdm(range(self.tto_steps), desc="TTO")
        initial_loss = None
        with torch.enable_grad():
            for _ in pbar:  # non-learning based optimization steps
                optimizer.zero_grad()

                recon_param = (
                    self.graspcvae.cvae.inference(
                        n=obj_pts.size(0), c=obj_features, z=z
                    )
                    .clone()
                    .detach()
                )  # Input: (B, 3, N), Output: (B, N_MANO_PARAMS)

                recon_xyz, recon_joints = self.affine_mano(
                    recon_param[:, :18],
                    recon_param[:, 18:28],
                    recon_param[:, 28:31],
                    rot_6d=recon_param[:, 31:37],
                )
                recon_anchors = self.affine_mano.get_anchors(recon_xyz)  # [B,32,3]

                # calculate cmap from current hand
                obj_nn_dist_affordance, _ = get_NN(
                    obj_pts, recon_xyz
                )  # Input: obj: (B, N, 3), hand: (B, 778, 3), Output: (B, N)
                cmap_affordance = get_pseudo_cmap(obj_nn_dist_affordance)  # [B,3000]

                # predict target cmap by ContactNet
                recon_cmap = self.contactnet(
                    obj_pts.permute(0, 2, 1), recon_xyz.permute(0, 2, 1).contiguous()
                )  # [B,3000]
                recon_cmap = (
                    recon_cmap / torch.max(recon_cmap, dim=1, keepdim=True)[0]
                ).detach()

                faces = self.affine_mano.faces.expand(B, -1, -1)
                penetr_loss, consistency_loss, contact_loss = TTT_loss(
                    recon_xyz,
                    faces,
                    obj_pts.contiguous(),  # [B, N, 3]
                    cmap_affordance,
                    recon_cmap,
                )
                loss = 1 * contact_loss + 1 * consistency_loss + 5 * penetr_loss
                if initial_loss is None:
                    initial_loss = f"Loss: {loss.item():.4f} Contact: {contact_loss.item():.4f} Consistency: {consistency_loss.item():.4f} Penetr: {penetr_loss.item():.4f}"
                pbar.set_description(
                    f"Loss: {loss.item():.4f} Contact: {contact_loss.item():.4f} Consistency: {consistency_loss.item():.4f} Penetr: {penetr_loss.item():.4f}"
                )
                loss.backward()

                # print(f"recon_param.grad: {recon_param.grad} / requires_grad: {recon_param.requires_grad}") #if recon_param.grad is not None: #print(f"params grads norms: {torch.norm(recon_param.grad[:, :18]).mean(0)}, {torch.norm(recon_param.grad[:, 18:28]).mean(0)}, {torch.norm(recon_param.grad[:, 28:31]).mean(0)}, {torch.norm(recon_param.grad[:, 31]).mean(0)}") optimizer.step()
                scheduler.step()

        print("========INITIAL=========")
        print(initial_loss)
        print("========================")
        # Final verts/joints:
        recon_xyz, recon_joints = self.affine_mano(
            recon_param[:, :18],
            recon_param[:, 18:28],
            recon_param[:, 28:31],
            rot_6d=recon_param[:, 31:37],
        )
        recon_anchors = self.affine_mano.get_anchors(recon_xyz)  # [B,32,3]
        return recon_xyz, recon_joints, recon_anchors, recon_param
