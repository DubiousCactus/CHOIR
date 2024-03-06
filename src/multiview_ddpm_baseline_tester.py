#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2024 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.


from typing import Dict, List, Optional, Tuple, Union

import open3d.io as o3dio
import pyvista as pv
import torch
from trimesh import Trimesh

from src.multiview_tester import MultiViewTester
from utils import to_cuda, to_cuda_
from utils.dataset import fetch_gaussian_params_from_CHOIR


class MultiViewDDPMBaselineTester(MultiViewTester):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conditional = kwargs.get("conditional", False)
        self._data_loader.dataset.set_observations_number(1)
        self._use_deltas = self._data_loader.dataset.use_deltas
        self._full_choir = kwargs.get("full_choir", False)
        self._is_baseline = True
        self._model_contacts = kwargs.get("model_contacts", False)
        if self._model_contacts:
            self._model.backbone.set_anchor_indices(
                self._data_loader.dataset.anchor_indices
            )

        self._model.set_dataset_stats(self._data_loader.dataset)
        # Because I infer the shape of the model from the data, I need to
        # run the model's forward pass once before calling .generate()
        print("[*] Running the model's forward pass once...")
        with torch.no_grad():
            samples, labels, _ = to_cuda_(next(iter(self._data_loader)))
            x = (
                torch.cat(
                    (
                        labels["rescaled_ref_pts"][:, -1],
                        labels["joints"][:, -1],
                        labels["anchors"][:, -1],
                    ),
                    dim=-2,  # Concat along the keypoints and not their dimensionality
                )
                if self._full_choir  # full_hand_object_pair
                else torch.cat(
                    (labels["joints"][:, -1], labels["anchors"][:, -1]), dim=-2
                )
            )
            y = (
                torch.cat(
                    (
                        samples["rescaled_ref_pts"],
                        samples["joints"],
                        samples["anchors"],
                    ),
                    dim=-2,
                )
                if self.conditional
                else None
            )
            kwargs = {"x": x, "y": y}
            if self._model_contacts:
                kwargs["contacts"] = fetch_gaussian_params_from_CHOIR(
                    labels["choir"].squeeze(1),
                    self._data_loader.dataset.anchor_indices,
                    n_repeats=self._data_loader.dataset.bps_dim // 32,
                    n_anchors=32,
                    choir_includes_obj=True,
                )[:, :, 0].squeeze(1)

            self._model(**kwargs)
        print("[+] Done!")

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
        samples, labels, mesh_pths = batch  # type: ignore
        mesh_pths = list(mesh_pths[-1])  # Now we have a list of B entries.
        random_idx = 7
        mesh_obj = o3dio.read_triangle_mesh(mesh_pths[random_idx])
        mesh_obj.translate(-mesh_obj.get_center())
        mano_params_gt = {
            "pose": labels["theta"],
            "beta": labels["beta"],
            "rot_6d": labels["rot"],
            "trans": labels["trans"],
        }
        mano_params_gt = {k: v[:, -1] for k, v in mano_params_gt.items()}
        verts_gt, joints_gt = self._affine_mano(
            mano_params_gt["pose"],
            mano_params_gt["beta"],
            mano_params_gt["trans"],
            rot_6d=mano_params_gt["rot_6d"],
        )
        print(verts_gt.shape, verts_gt[random_idx].shape)
        import open3d

        hand_mesh = open3d.geometry.TriangleMesh(
            vertices=open3d.utility.Vector3dVector(
                verts_gt[random_idx].detach().cpu().numpy()
            ),
            triangles=open3d.utility.Vector3iVector(
                self._affine_mano.faces.detach().cpu().numpy()
            ),
        )
        hand_mesh.compute_vertex_normals()
        print(hand_mesh)
        t_mesh_obj = Trimesh(vertices=mesh_obj.vertices, faces=mesh_obj.triangles)
        # We need a cone mesh and an arrow for the normal. The cone has an angle of 30 degrees and
        # is along the normal, of about 2cm length:
        # 740, 750
        mano_vertex_idx_1, mano_vertex_idx_2 = 324, 750
        normal1 = hand_mesh.vertex_normals[mano_vertex_idx_1]
        normal2 = hand_mesh.vertex_normals[mano_vertex_idx_2]
        # Make it a unit vector:
        import numpy as np

        normal1 /= np.linalg.norm(normal1)
        normal2 /= np.linalg.norm(normal2)
        normal_arrow1 = pv.Arrow(
            start=hand_mesh.vertices[mano_vertex_idx_1],
            direction=normal1,
            scale=0.02,
            tip_length=0.05,
            tip_radius=0.05,
            shaft_radius=0.02,
        )
        normal_arrow2 = pv.Arrow(
            start=hand_mesh.vertices[mano_vertex_idx_2],
            direction=normal2,
            scale=0.02,
            tip_length=0.05,
            tip_radius=0.05,
            shaft_radius=0.02,
        )
        # Let's make the cone along the normal, at the vertex position:
        cone1 = pv.Cone(
            direction=-normal1,
            height=0.02,
            radius=0.005,
            resolution=100,
            capping=True,
            center=hand_mesh.vertices[mano_vertex_idx_1] + (0.005 * normal1),
        )
        cone2 = pv.Cone(
            direction=-normal2,
            height=0.02,
            radius=0.005,
            resolution=100,
            capping=True,
            center=hand_mesh.vertices[mano_vertex_idx_2] + (0.005 * normal2),
        )

        # Let's use pyvista to visualize hand and object meshes:
        p = pv.Plotter()
        p.add_mesh(
            Trimesh(hand_mesh.vertices, hand_mesh.triangles),
            color="lightblue",
            show_edges=True,
            opacity=1.0,
            smooth_shading=True,
        )
        p.add_mesh(
            t_mesh_obj,
            color="orange",
            show_edges=False,
            opacity=1.0,
            smooth_shading=True,
        )
        p.add_mesh(normal_arrow1, color="purple")
        p.add_mesh(cone1, color="purple", opacity=0.5)
        p.add_mesh(normal_arrow2, color="green")
        p.add_mesh(cone2, color="green", opacity=0.5)
        p.show()
        raise Exception
        # visualize_model_predictions_with_multiple_views(
        # self._model,
        # batch,
        # epoch,
        # bps_dim=self._bps_dim,
        # bps=self._bps,
        # anchor_indices=self._anchor_indices,
        # remap_bps_distances=self._remap_bps_distances,
        # exponential_map_w=self._exponential_map_w,
        # dataset=self._data_loader.dataset.name,
        # theta_dim=self._data_loader.dataset.theta_dim,
        # use_deltas=self._use_deltas,
        # conditional=self.conditional,
        # method="baseline",
        # )  # User implementation goes here (utils/training.py)

    # @to_cuda
    def _inference(
        self,
        samples,
        labels,
        max_observations: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Training or validation procedure for one batch. We want to keep the code DRY and avoid
        making mistakes, so write this code only once at the cost of many function calls!
        Args:
            batch: The batch to process.
        Returns:
            torch.Tensor: The loss for the batch.
        """
        max_observations = max_observations or samples["choir"].shape[1]
        kp, contacts = self._model.generate(
            1,
            y=torch.cat(
                (
                    samples["rescaled_ref_pts"][:, :max_observations],
                    samples["joints"][:, :max_observations],
                    samples["anchors"][:, :max_observations],
                ),
                dim=-2,
            )
            if self.conditional
            else None,
        )
        # Only use 1 sample for now. TODO: use more samples and average?
        kp, contacts = kp.squeeze(1), contacts.squeeze(1)
        return {"hand_keypoints": kp, "contacts": contacts}
