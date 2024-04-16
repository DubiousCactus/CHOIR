#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2024 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

import os
from typing import Dict

import numpy as np
from trimesh import Trimesh


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
        self.scene = sp.Scene()
        self.n_frames = 0
        self.main = self.scene.create_canvas_3d(
            width=width,
            height=height,
            shading=sp.Shading(bg_color=np.array([255 / 255, 252 / 255, 251 / 255])),
        )
        self.colors = sp.Colors

    def meshes_to_sp(self, meshes: Dict[str, Trimesh]):
        sp_meshes = []
        for mesh_name, mesh in meshes.items():
            colors = {
                "hand": np.array([181 / 255, 144 / 255, 191 / 255]),
                "object": np.array([137 / 255, 189 / 255, 223 / 255]),
            }
            params = {
                "vertices": mesh.vertices.astype(np.float32),
                # "normals": mesh.vertex_normals.astype(np.float32),
                "triangles": mesh.faces.astype(np.int32),
                # "colors": mesh.visual.vertex_colors.astype(np.float32)[..., :3] / 255.0,
                colors: colors[mesh_name]
                if mesh_name in colors
                else np.array([0.5, 0.5, 0.5]),
            }
            # params = {'vertices' : m.v.astype(np.float32), 'triangles' : m.f, 'colors' : m.vc.astype(np.float32)}
            # sp_m = sp.Mesh()
            sp_m = self.scene.create_mesh(layer_id=mesh_name)
            sp_m.add_mesh_without_normals(**params)
            if mesh_name == "ground_mesh":
                sp_m.double_sided = True
            sp_meshes.append(sp_m)
        return sp_meshes

    def add_frame(self, meshes: Dict[str, Trimesh]):
        meshes_list = self.meshes_to_sp(meshes)
        if not hasattr(self, "focus_point"):
            self.focus_point = list(meshes.values())[0].centroid
            # center = self.focus_point
            # center[2] = 4
            # rotation = sp.Transforms.rotation_about_z(0)
            # self.camera = sp.Camera(center=center, rotation=rotation, fov_y_degrees=30.0)

        main_frame = self.main.create_frame(focus_point=self.focus_point)
        for i, m in enumerate(meshes_list):
            # self.main.set_layer_settings({layer_names[i]:{}})
            main_frame.add_mesh(m)
        self.n_frames += 1

    def save_animation(self, sp_anim_name):
        self.scene.link_canvas_events(self.main)
        self.scene.save_as_html(sp_anim_name, title=os.path.basename(sp_anim_name))
