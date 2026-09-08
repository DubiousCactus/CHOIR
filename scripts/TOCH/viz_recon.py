#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2024 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Visualize reconstruction results
"""

# import open3d as o3d
import pyvista as pv

input_hand_pth = "recon_results/input_hand_0.ply"
recon_hand_pth = "recon_results/hand_0.ply"
object_pth = "recon_results/object_0.ply"


# def main_o3d():
#     input_hand = o3d.io.read_triangle_mesh(input_hand_pth)
#     input_hand.compute_vertex_normals()
#     input_hand.paint_uniform_color([0.9, 0.1, 0.1])  # Red-ish
#     recon_hand = o3d.io.read_triangle_mesh(recon_hand_pth)
#     recon_hand.compute_vertex_normals()
#     recon_hand.paint_uniform_color([0.1, 0.1, 0.9])  # Blue-ish
#     object = o3d.io.read_triangle_mesh(object_pth)
#     object.compute_vertex_normals()
#
#     o3d.visualization.draw_geometries([input_hand, recon_hand, object])


def main_pv():
    input_hand = pv.read(input_hand_pth)
    # input_hand.compute_normals(inplace=True)
    recon_hand = pv.read(recon_hand_pth)
    # recon_hand.compute_normals(inplace=True)
    obj = pv.read(object_pth)
    obj.compute_normals(inplace=True)
    plotter = pv.Plotter()
    # plotter.add_mesh(input_hand, color="red")
    plotter.add_mesh(recon_hand, color="#2265C9")  # [0.1, 0.1, 0.9])
    plotter.add_mesh(obj, color="red")
    # plotter.show()
    plotter.export_html("test.html")


if __name__ == "__main__":
    main_pv()
