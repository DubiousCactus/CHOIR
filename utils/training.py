#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Training utilities. This is a good place for your code that is used in training (i.e. custom loss
function, visualization code, etc.)
"""

from typing import Optional, Tuple

import torch
import tqdm
from manotorch.anchorlayer import AnchorLayer

from model.affine_mano import AffineMANO
from src.losses.hoi import DualHOILoss


def optimize_pose_pca_from_choir(
    choir: torch.Tensor,
    bps_dim: int,
    x_mean: torch.Tensor,
    x_scalar: float,
    hand_contacts: Optional[torch.Tensor] = None,
    loss_thresh: float = 1e-12,
    max_iterations=8000,
    objective="anchors",
    initial_params=None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ncomps = 15
    affine_mano = AffineMANO(ncomps).cuda()
    anchor_layer = AnchorLayer(anchor_root="vendor/manotorch/assets/anchor").cuda()
    choir_loss = DualHOILoss(bps_dim).cuda()
    B = choir.shape[0]
    if initial_params is None:
        fingers_pose = (torch.rand((1, ncomps + 3))).cuda().requires_grad_(True)
        shape = (torch.rand((1, 10))).cuda().requires_grad_(True)
        rot_6d = torch.rand((1, 6)).cuda().requires_grad_(True)
        trans = (torch.rand((1, 3)) * 0.001).cuda().requires_grad_(True)
    else:
        fingers_pose = initial_params["pose"].cuda().requires_grad_(True)
        shape = initial_params["shape"].cuda().requires_grad_(True)
        rot_6d = initial_params["rot_6d"].cuda().requires_grad_(True)
        trans = initial_params["trans"].cuda().requires_grad_(True)
    parameters = {
        "rot": rot_6d,
        "trans": trans,
        "fingers_pose": fingers_pose,
        "shape": shape,
    }
    params = [{"params": parameters.values()}]

    optimizer = torch.optim.Adam(params, lr=1e-5 if objective == "contacts" else 3e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.9)
    alpha, beta = 1.0, 0.0
    alpha_lower_bound, beta_upper_bound = 0.5, 0.5
    proc_bar = tqdm.tqdm(range(max_iterations))

    prev_loss = float("inf")
    anchors = torch.zeros((B, 32, 3)).cuda()

    for i, _ in enumerate(proc_bar):
        optimizer.zero_grad()
        verts, _ = affine_mano(fingers_pose, shape, rot_6d, trans)
        anchors = anchor_layer(verts)
        anchor_loss, contacts_loss = choir_loss(
            verts, anchors, choir, hand_contacts, x_mean, x_scalar
        )
        proc_bar.set_description(
            f"Anchors loss: {anchor_loss.item():.10f} / Contacts loss: {contacts_loss.item():.10f}"
        )
        if objective == "both":
            # Aneal alpha and beta so that we start with a focus on the anchors and then
            # move to the contacts, as i gets closer to max_iterations.
            alpha = max(alpha * (0.99 ** (i / max_iterations)), alpha_lower_bound)
            beta = min(beta * (1.11 ** (i / max_iterations)), beta_upper_bound)
            loss = alpha * anchor_loss + beta * contacts_loss
        elif objective == "contacts":
            loss = contacts_loss
        elif objective == "anchors":
            loss = anchor_loss
        else:
            raise ValueError(
                f"Unknown objective: {objective}. Must be one of 'both', 'contacts', 'anchors'"
            )
        if torch.abs(prev_loss - loss.detach()) < loss_thresh:
            break
        prev_loss = loss.detach()
        loss.backward()
        optimizer.step()
        scheduler.step()

    return (
        fingers_pose.detach(),
        shape.detach(),
        rot_6d.detach(),
        trans.detach(),
        anchors[0].unsqueeze(0),
    )
