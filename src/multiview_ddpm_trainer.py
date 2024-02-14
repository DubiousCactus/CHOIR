#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2024 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.


from typing import Dict, List, Tuple, Union

import torch
from ema_pytorch import EMA

from src.base_trainer import BaseTrainer
from utils import DebugMetaclass, to_cuda
from utils.visualization import visualize_model_predictions_with_multiple_views


class MultiViewDDPMTrainer(BaseTrainer, metaclass=DebugMetaclass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conditional = kwargs.get("conditional", False)
        self._use_deltas = self._train_loader.dataset.use_deltas
        self._full_choir = kwargs.get("full_choir", False)
        self._model_contacts = kwargs.get("model_contacts", False)
        if self._model_contacts:
            self._model.backbone.set_anchor_indices(
                self._train_loader.dataset.anchor_indices
            )
            self._model.set_dataset_stats(self._train_loader.dataset)
        self._ema = EMA(
            self._model, beta=0.9999, update_after_step=100, update_every=10
        )

    @to_cuda
    def _visualize(
        self,
        batch: Union[Tuple, List, torch.Tensor],
        epoch: int,
    ) -> None:
        """Visualize the model predictions.
        Args:
            batch: The batch to process.
            epoch: The current epoch.
        """
        samples, labels, _ = batch
        with torch.no_grad():
            # Initialize the EMA model with a forward pass before generation
            self._ema.ema_model(
                labels["choir"][:, -1]
                if self._model.embed_full_choir
                else labels["choir"][:, -1][..., -1].unsqueeze(-1),
                samples["choir"] if self.conditional else None,
            )
        visualize_model_predictions_with_multiple_views(
            self._ema.ema_model,
            (samples, labels, None),
            epoch,
            bps_dim=self._bps_dim,
            bps=self._bps,
            anchor_indices=self._anchor_indices,
            remap_bps_distances=self._remap_bps_distances,
            exponential_map_w=self._exponential_map_w,
            dataset=self._train_loader.dataset.name,
            theta_dim=self._train_loader.dataset.theta_dim,
            use_deltas=self._use_deltas,
            conditional=self.conditional,
            method="ddpm",
        )  # User implementation goes here (utils/training.py)

    @to_cuda
    def _train_val_iteration(
        self,
        batch: Union[Tuple, List, torch.Tensor],
        validation: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Training or validation procedure for one batch. We want to keep the code DRY and avoid
        making mistakes, so write this code only once at the cost of many function calls!
        Args:
            batch: The batch to process.
        Returns:
            torch.Tensor: The loss for the batch.
        """
        samples, labels, _ = batch

        # ========= Integration Testing ==========
        # Let's test the 3D Gaussian parameters encoded in the label's CHOIR:
        """
        with colorize_prints(project_conf.ANSI_COLORS["yellow"]):
            print(
                "[Integration Testing] Testing 3D Gaussian parameters",
            )
            last_frame_choir = labels["choir"][0, -1].unsqueeze(0)
            bps_dim = self._train_loader.dataset.bps_dim
            repeated_gaussian_params = fetch_gaussian_params_from_CHOIR(
                last_frame_choir,
                self._train_loader.dataset.anchor_indices,
                n_repeats=bps_dim // 32,
                n_anchors=32,
                choir_includes_obj=True,
            )
            print(
                f"Extracted Gaussian parameters shape: {repeated_gaussian_params.shape}"
            )
            # Simulate some basic pooling:
            gaussian_params = repeated_gaussian_params.mean(dim=-2)
            # gaussian_params = repeated_gaussian_params[..., 0, :]
            print(f"Pooled Gaussian parameters shape: {gaussian_params.shape}")
            mus = gaussian_params[..., :3]
            lower_tril_covs = gaussian_params[..., 3:]
            print(
                f"Mus shape: {mus.shape}. Lower tril. covs shape: {lower_tril_covs.shape}"
            )
            recovered_covs = lower_tril_cholesky_to_covmat(lower_tril_covs)[0]
            print(f"Recovered covs shape: {recovered_covs.shape}")
            # TODO: Remove the hard-coded scaling (see dataset/contactpose.py around line 540)
            choleksy_gaussian_params = torch.cat(
                (mus.squeeze(0) / 100, recovered_covs.view(-1, 9) / 1000), dim=-1
            ).cpu()
            # Visualize the Gaussian parameters
            from open3d import geometry as o3dg
            from open3d import io as o3dio
            from open3d import utility as o3du
            from open3d import visualization as o3dv
            import numpy as np

            theta, beta, rot, trans = (
                labels["theta"][0, -1].unsqueeze(0),
                labels["beta"][0, -1].unsqueeze(0),
                labels["rot"][0, -1].unsqueeze(0),
                labels["trans"][0, -1].unsqueeze(0),
            )
            print(f"Theta shape: {theta.shape}. Beta shape: {beta.shape}")
            affine_mano: AffineMANO = AffineMANO(for_contactpose=True).to(theta.device)  # type: ignore
            gt_verts, _ = affine_mano(theta, beta, rot, trans)
            faces = affine_mano.faces.detach().cpu().numpy()
            gt_anchors = affine_mano.get_anchors(gt_verts).detach().cpu()  # .numpy()
            gt_hand_mesh = o3dg.TriangleMesh()
            gt_hand_mesh.vertices = o3du.Vector3dVector(
                gt_verts[0].detach().cpu().numpy()
            )
            gt_hand_mesh.triangles = o3du.Vector3iVector(faces)
            colours = np.zeros_like(gt_verts[0].detach().cpu().numpy())
            # Visualize contacts by colouring the vertices
            colours[:, 0] = 0.00
            colours[:, 1] = 0.58
            colours[:, 2] = 0.66
            gt_hand_mesh.vertex_colors = o3du.Vector3dVector(colours)

            obj_mesh_path = batch[-1][0][
                0
            ]  # First batch element (why is the batch a list of 1 tuple of batch elements??)
            obj_mesh = o3dio.read_triangle_mesh(obj_mesh_path)
            obj_center = torch.from_numpy(obj_mesh.get_center())
            obj_mesh.translate(-obj_center)

            o3dv.draw_geometries([obj_mesh, gt_hand_mesh])
            print("====== Visualizing ground-truth 3D Gaussians on hand mesh ======")
            gt_gaussian_params = labels["contact_gaussians"][0, -1].cpu()
            print(f"GT Gaussian parameters shape: {gt_gaussian_params.shape}")
            visualize_3D_gaussians_on_hand_mesh(
                gt_hand_mesh,
                obj_mesh,
                gt_gaussian_params,
                base_unit=self._train_loader.dataset.base_unit,
                anchors=gt_anchors[0],
            )

            print(
                "====== Reconstructing contacts from ground-truth 3D Gaussians ======"
            )
            visualize_hand_contacts_from_3D_gaussians(
                gt_hand_mesh,
                gt_gaussian_params,
                gt_anchors[0],
            )

            print("====== Visualizing recomposed 3D Gaussians on hand mesh ======")
            gt_hand_mesh.vertex_colors = o3du.Vector3dVector(colours)
            visualize_3D_gaussians_on_hand_mesh(
                gt_hand_mesh,
                obj_mesh,
                choleksy_gaussian_params,
                base_unit=self._train_loader.dataset.base_unit,
                anchors=gt_anchors[0],
            )

            print("====== Reconstructing contacts from recomposed 3D Gaussians ======")
            visualize_hand_contacts_from_3D_gaussians(
                gt_hand_mesh,
                choleksy_gaussian_params,
                gt_anchors[0],
                # gt_contacts=gt_vertex_contacts,
            )
        # ========================================
        """
        # Take the last frame (-1):
        # TODO: Refactor this massive crap:
        x = (
            labels["choir"][:, -1]
            if self._full_choir
            else (
                labels["choir"][:, -1][..., -1].unsqueeze(-1)
                if not self._model_contacts
                else (
                    labels["choir"][:, -1]
                    if self._model.object_in_encoder
                    else labels["choir"][:, -1][..., 1:]
                )
            )
        )
        y = samples["choir"] if self.conditional else None

        if not self._use_deltas:
            y_hat = self._model(x, y)
        else:
            raise NotImplementedError(
                "Have to scrap embed_full_choir in DiffusionModel. I tried and it's not better."
            )
        losses = self._training_loss(None, None, y_hat)
        # dict: {"udf_mse": torch.Tensor, "contacts_mse": torch.Tensor}
        if validation:
            ema_y_hat = self._ema.ema_model(x, y)
            ema_loss = self._training_loss(None, None, ema_y_hat)
            losses["ema"] = sum([v for v in ema_loss.values()])

        loss = sum([v for k, v in losses.items() if k != "ema"])
        # Reweight the losses to plot the *true* loss:
        losses["contacts_mse"] /= self._training_loss.contacts_weight
        return loss, losses
