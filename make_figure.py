import pickle
from typing import Optional, Tuple

import blosc2
import numpy as np
import open3d as o3d
import pyvista as pv
import torch
import trimesh
from manotorch.anchorlayer import AnchorLayer
from manotorch.manolayer import ManoLayer, MANOOutput
from pytorch3d.transforms import Transform3d
from pytorch3d.transforms.rotation_conversions import (
    rotation_6d_to_matrix,
)
from tqdm import tqdm

fpath = "data/ContactPose_preprocessed/samples_and_labels/perturbed-2__10000-obj-pts_right-hand_1024-bps_object-centered_none-rescaled_exponential_mapped-5.0_random-anchors_train/mug_1_use_right_aug-0/sample_000000.pkl"

fpath = "data/ContactPose_preprocessed/samples_and_labels/perturbed-2__10000-obj-pts_right-hand_1024-bps_object-centered_none-rescaled_exponential_mapped-5.0_random-anchors_test/knife_9_handoff_right_aug-0/sample_000000.pkl"


def transform_verts(
    verts: torch.Tensor, rot6d: torch.Tensor, t: torch.Tensor, apply_inverse_rot: bool
) -> torch.Tensor:
    R = rotation_6d_to_matrix(rot6d)
    if apply_inverse_rot:
        # NOTE: This is the inverse of the transform we want to apply to the MANO model. It works
        # for the code given in ContactPose because they're applying the inverse transform to the
        # points with np.dot(transform, points.T).
        inv_r = Transform3d().rotate(R).inverse().to(verts.device)
        transform = (
            Transform3d(device=verts.device)
            .compose(inv_r)
            .translate(t)
            .to(verts.device)
        )
    else:
        transform = (
            Transform3d(device=verts.device).rotate(R).translate(t).to(verts.device)
        )
    return transform.transform_points(verts)


class AffineMANO(torch.nn.Module):
    def __init__(
        self,
        ncomps: int = 15,
        flat_hand_mean: bool = False,
        for_contactpose: bool = False,
        for_oakink: bool = False,
    ):
        super().__init__()
        # MANO is shipped with 15 components but you can use less.
        self.for_oakink = for_oakink
        if for_oakink:
            # rot_mode=axisang, side=right, use_pca=False, flat_hand_mean=True
            self.mano_layer = ManoLayer(
                center_idx=9,
                mano_assets_root="vendor/manotorch/assets/mano",
            )
        else:
            self.mano_layer = ManoLayer(
                mano_assets_root="vendor/manotorch/assets/mano",
                use_pca=True,
                flat_hand_mean=flat_hand_mean,
                ncomps=ncomps,
            )
        self.anchor_layer = AnchorLayer(anchor_root="vendor/manotorch/assets/anchor")
        self._for_contactpose = for_contactpose

    def forward(
        self,
        pose: torch.Tensor,
        shape: torch.Tensor,
        trans: torch.Tensor,
        rot_6d: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mano_output: MANOOutput = self.mano_layer(pose, shape)
        verts = mano_output.verts
        joints = mano_output.joints
        if self.for_oakink:
            return verts + trans[:, None, :], joints + trans[:, None, :]
        else:
            return transform_verts(
                verts, rot_6d, trans, apply_inverse_rot=self._for_contactpose
            ), transform_verts(
                joints, rot_6d, trans, apply_inverse_rot=self._for_contactpose
            )

    @property
    def faces(self) -> torch.Tensor:
        return self.mano_layer.th_faces  # type: ignore

    @property
    def closed_faces(self) -> torch.Tensor:
        return self.mano_layer.get_mano_closed_faces()  # type: ignore

    def get_anchors(self, verts: torch.Tensor) -> torch.Tensor:
        return self.anchor_layer(verts)


with open(fpath, "rb") as f:
    compressed_pickle = f.read()
    sample, label, mesh_pth = pickle.loads(blosc2.decompress(compressed_pickle))
    affine_mano = AffineMANO(for_contactpose=True)
    theta, beta, rot_6d, trans = (
        torch.from_numpy(label[5]),
        torch.from_numpy(label[6]),
        torch.from_numpy(label[7]),
        torch.from_numpy(label[8]),
    )
    hand_verts, hand_joints = affine_mano(
        theta[None], beta[None], trans[None], rot_6d=rot_6d[None]
    )

    t_hand_mesh = trimesh.Trimesh(
        vertices=hand_verts[0].detach().cpu().numpy(),
        faces=affine_mano.closed_faces,
    )
    t_hand_mesh.fix_normals()

    obj_mesh = o3d.io.read_triangle_mesh(mesh_pth)
    obj_mesh.translate(-obj_mesh.get_center())
    t_obj_mesh = trimesh.Trimesh(
        vertices=obj_mesh.vertices,
        faces=obj_mesh.triangles,
        process=False,
        validate=False,
    )
    t_obj_mesh.fix_normals()
    normals_w_roots = torch.cat(
        (
            torch.from_numpy(np.asarray(t_obj_mesh.vertices).copy()).type(
                dtype=torch.float32
            ),
            torch.from_numpy(np.asarray(t_obj_mesh.vertex_normals).copy()).type(
                dtype=torch.float32
            ),
        ),
        dim=-1,
    )

    plotter = pv.Plotter()
    plotter.add_mesh(pv.wrap(t_hand_mesh), show_edges=False)
    plotter.add_mesh(pv.wrap(t_obj_mesh), color="red", show_edges=False)

    # Now draw rays from the object normals, with either one of these 2 colours:
    # 1) green if the ray intersects with the hand surface.
    # 2) purple if the ray doesn't intersect.
    for i in tqdm(range(normals_w_roots.shape[0])):
        v, n = normals_w_roots[i, :3], normals_w_roots[i, 3:]
        intersector = trimesh.ray.ray_triangle.RayMeshIntersector(t_hand_mesh)
        locations, index_rays, index_triangles = intersector.intersects_location(
            ray_origins=v[None], ray_directions=n[None]
        )
        if len(locations) > 0:
            if i % 10 != 0:
                continue
            # We just want the first intersection:
            loc = locations[0]
        else:
            if i % 100 != 0:
                continue
            loc = v + 0.03 * n
        plotter.add_mesh(
            pv.Line(pointa=v, pointb=loc),
            color="green" if len(locations) > 0 else "purple",
            line_width=4,
        )

    print("[*] Exporting the figure to hand.html...")
    plotter.export_html("hand.html")
