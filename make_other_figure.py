import os
import pickle
from typing import Dict, Optional, Tuple

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
from trimesh import Trimesh

colors = {
    "hand": np.array([181 / 255, 144 / 255, 191 / 255]),
    "object": np.array([137 / 255, 189 / 255, 223 / 255]),
}

# Colour palette:
palette = [
    {
        "name": "African Violet",
        "hex": "b590bf",
        "rgb": [181, 144, 191],
        "cmyk": [5, 25, 0, 25],
        "hsb": [287, 25, 75],
        "hsl": [287, 27, 66],
        "lab": [65, 22, -19],
    },
    {
        "name": "UCLA Blue",
        "hex": "2973b0",
        "rgb": [41, 115, 176],
        "cmyk": [77, 35, 0, 31],
        "hsb": [207, 77, 69],
        "hsl": [207, 62, 43],
        "lab": [47, -1, -39],
    },
    {
        "name": "Snow",
        "hex": "fffcfb",
        "rgb": [255, 252, 251],
        "cmyk": [0, 1, 2, 0],
        "hsb": [15, 2, 100],
        "hsl": [15, 100, 99],
        "lab": [99, 1, 1],
    },
    {
        "name": "African Violet",
        "hex": "c179bc",
        "rgb": [193, 121, 188],
        "cmyk": [0, 37, 3, 24],
        "hsb": [304, 37, 76],
        "hsl": [304, 37, 62],
        "lab": [60, 38, -24],
    },
    {
        "name": "Carolina blue",
        "hex": "89bddf",
        "rgb": [137, 189, 223],
        "cmyk": [39, 15, 0, 13],
        "hsb": [204, 39, 87],
        "hsl": [204, 57, 71],
        "lab": [74, -9, -22],
    },
]


class ScenePicAnim:
    def __init__(
        self,
        width=1600,
        height=1600,
    ):
        super().__init__()
        try:
            import scenepic as sp
        except:
            raise Exception(
                "scenepic not installed. Some visualization functions will not work. (I know it's not available on Apple Silicon :("
            )
        pv.start_xvfb()
        self.scene = sp.Scene()
        self.n_frames = 0
        self.main = self.scene.create_canvas_3d(
            width=width,
            height=height,
            shading=sp.Shading(bg_color=np.array([255 / 255, 252 / 255, 251 / 255])),
        )
        self.colors = sp.Colors
        self.positions = dict()

    def meshes_to_sp(self, meshes: Dict[str, Trimesh], reuse: Optional[str] = None):
        sp_meshes = []
        for mesh_name, mesh in meshes.items():
            params = {
                "vertices": mesh.vertices.astype(np.float32),
                # "normals": mesh.vertex_normals.astype(np.float32),
                "triangles": mesh.faces.astype(np.int32),
                # "colors": mesh.visual.vertex_colors.astype(np.float32)[..., :3] / 255.0,
                "colors": np.expand_dims(
                    colors[mesh_name]
                    if mesh_name in colors
                    else np.array([0.5, 0.5, 0.5]),
                    axis=0,
                ).repeat(mesh.vertices.shape[0], axis=0),
            }
            # params = {'vertices' : m.v.astype(np.float32), 'triangles' : m.f, 'colors' : m.vc.astype(np.float32)}
            # sp_m = sp.Mesh()
            if reuse is not None and reuse == mesh_name and mesh_name in self.positions:
                sp_m = self.scene.update_mesh_positions(
                    self.positions[mesh_name][0], self.positions[mesh_name][1].copy()
                )
            else:
                sp_m = self.scene.create_mesh(layer_id=mesh_name)
                sp_m.add_mesh_without_normals(**params)
                self.positions[mesh_name] = (
                    sp_m.mesh_id,
                    mesh.vertices.astype(np.float32),
                )
            if mesh_name == "ground_mesh":
                sp_m.double_sided = True
            sp_meshes.append(sp_m)
        return sp_meshes

    def add_frame(self, meshes: Dict[str, Trimesh], reuse: Optional[str] = None):
        meshes_list = self.meshes_to_sp(meshes, reuse)
        if not hasattr(self, "focus_point"):
            self.focus_point = np.array(
                [0, 0, 0]
            )  # list(meshes.values())[0].center_mass
            # center = self.focus_point
            # center[2] = 4
            # rotation = sp.Transforms.rotation_about_z(0)
            # self.camera = sp.Camera(center=center, rotation=rotation, fov_y_degrees=30.0)

        # main_frame = self.main.create_frame(focus_point=self.focus_point)
        main_frame = self.main.create_frame()
        for i, m in enumerate(meshes_list):
            # self.main.set_layer_settings({layer_names[i]:{}})
            main_frame.add_mesh(m)
        self.n_frames += 1

    def save_animation(self, sp_anim_name):
        print(f"[*] Saving animation as {sp_anim_name}...")
        self.scene.link_canvas_events(self.main)
        self.scene.save_as_html(sp_anim_name, title=os.path.basename(sp_anim_name))


fpath = "data/ContactPose_preprocessed/samples_and_labels/perturbed-2__10000-obj-pts_right-hand_1024-bps_object-centered_none-rescaled_exponential_mapped-5.0_random-anchors_train/mug_1_use_right_aug-0/sample_000000.pkl"

fpath = "data/ContactPose_preprocessed/samples_and_labels/perturbed-2__10000-obj-pts_right-hand_1024-bps_object-centered_none-rescaled_exponential_mapped-5.0_random-anchors_test/knife_9_handoff_right_aug-0/sample_000000.pkl"

knives_path_1 = "data/ContactPose_preprocessed/samples_and_labels/perturbed-2__10000-obj-pts_right-hand_1024-bps_object-centered_none-rescaled_exponential_mapped-5.0_random-anchors_test/"
knives_path_2 = "data/ContactPose_preprocessed/samples_and_labels/perturbed-2__10000-obj-pts_right-hand_1024-bps_object-centered_none-rescaled_exponential_mapped-5.0_random-anchors_train/"
knives_path_3 = "data/ContactPose_preprocessed/samples_and_labels/perturbed-2__10000-obj-pts_right-hand_1024-bps_object-centered_none-rescaled_exponential_mapped-5.0_random-anchors_val/"


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


#
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
# plotter = pv.Plotter()
# plotter.add_mesh(pv.wrap(t_hand_mesh), show_edges=False)
# plotter.add_mesh(pv.wrap(t_obj_mesh), color="red", show_edges=False)
#
# print("[*] Exporting the figure to hand.html...")
# plotter.export_html("hand.html")

scene = ScenePicAnim(width=800, height=450)
theta, beta, rot_6d, trans = (
    torch.from_numpy(label[5]),
    torch.from_numpy(label[6]),
    torch.from_numpy(label[7]),
    torch.from_numpy(label[8]),
)
target_rot = torch.randn_like(rot_6d)
target_trans = torch.clamp(torch.randn_like(trans) * 3, -0.3, 0.3)
print(target_trans)
target_theta = torch.randn_like(theta)
for i in range(6 * 30):
    if i % 15 == 0:
        # target_rot = torch.clamp(torch.randn_like(rot_6d) * 1, -0.1, 0.1)
        target_trans = torch.clamp(torch.randn_like(trans) * 3, -0.3, 0.3)
        print(target_trans)
        target_theta = torch.randn_like(theta)
    rot_6d += (target_rot - rot_6d) * 0.1
    trans += (target_trans - trans) * 0.1
    theta += (target_theta - theta) * 0.1
    # theta = torch.clamp(theta, -1.0, 1.0)
    # rot_6d = torch.clamp(rot_6d, -1.0, 1.0)
    # trans = torch.clamp(trans, -1.0, 1.0)
    hand_verts, hand_joints = affine_mano(
        theta[None], beta[None], trans[None], rot_6d=rot_6d[None]
    )

    new_hand = trimesh.Trimesh(
        vertices=hand_verts[0].detach().cpu().numpy(),
        faces=affine_mano.closed_faces,
    )
    new_hand.fix_normals()
    scene.add_frame(
        {
            "object": t_obj_mesh,
            "hand": new_hand,
        },
        reuse="object",
    )
scene.save_animation("vast_space.html")

scene = ScenePicAnim(width=800, height=450)
dirs = [
    os.path.join(knives_path, dir)
    for knives_path in [knives_path_1, knives_path_2, knives_path_3]
    for dir in os.listdir(knives_path)
    if dir.startswith("knife")
]
for dir in dirs:
    print(f"using {os.path.basename(dir)}")
    path = os.path.join(dir, "sample_000000.pkl")
    with open(path, "rb") as f:
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
        scene.add_frame(
            {
                "object": t_obj_mesh,
                "hand": t_hand_mesh,
            },
            reuse="object",
        )
scene.save_animation("valid_space.html")
