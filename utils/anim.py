#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2024 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

import os
from typing import Dict, Optional

import numpy as np
import pyvista as pv
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
