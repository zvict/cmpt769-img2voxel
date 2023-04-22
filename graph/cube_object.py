import copy

import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt

import optimization
from mpl_toolkits.mplot3d import Axes3D

from main import rotate_point_cloud_align_z_axis


class CustomCube():
    def __init__(self, cube_triangle_meshes, nodes):
        self.nodes = nodes
        self.cube_triangle_meshes = cube_triangle_meshes
        self.cube_norms = []
        self.distinct_norms = {}  # key: plane_normals, value: best aligned norm on cube
        self.distinct_faces = {}
        # self.visualize_cube()
        self.get_distinct_norms()

    def visualize_cube(self):
        pcd = o3d.geometry.PointCloud()
        for n in self.nodes:
            new_points = np.concatenate((n.projected_points, np.asarray(pcd.points)), axis=0)
            pcd.points = o3d.utility.Vector3dVector(new_points)
        # set colors
        pcd.paint_uniform_color([0.5, 0.5, 0.5])
        o3d.visualization.draw_geometries([pcd, self.cube_triangle_meshes])

    def get_norms_from_cube_triangle_meshes(self):
        corners = np.asarray(self.cube_triangle_meshes.vertices)
        faces = np.array([
            [0, 1, 3, 2],  # bottom
            [0, 2, 6, 4],  # left
            [0, 1, 5, 4],  # front
            [1, 3, 7, 5],  # right
            [2, 3, 7, 6],  # back
            [4, 5, 7, 6],  # top
        ])

        # Compute the surface normals for each face of the cube
        normals = []
        for face in faces:
            edge1 = corners[face[1]] - corners[face[0]]
            edge2 = corners[face[2]] - corners[face[0]]
            normal = np.cross(edge1, edge2)
            normal = normal / np.linalg.norm(normal)
            normals.append(normal)
        return normals, faces

    def get_nearest_aligned_norm(self, node, faces):
        max_dot_product = 0
        best_norm = None
        face = None
        for idx, norm in enumerate(self.cube_norms):
            n1 = norm / np.linalg.norm(norm)
            n2 = node.plane_normals / np.linalg.norm(node.plane_normals)
            dot_product = np.dot(n1, n2)
            if dot_product > max_dot_product:
                max_dot_product = dot_product
                best_norm = norm
                face = faces[idx]
        return best_norm, face

    def get_distinct_norms(self):
        self.cube_norms, faces = self.get_norms_from_cube_triangle_meshes()
        # clear self.distinct_norms dict
        self.distinct_norms = {}
        self.distinct_faces = {}
        for n in self.nodes:
            nearest_norm, face = self.get_nearest_aligned_norm(n, faces)
            self.distinct_norms[n] = nearest_norm
            self.distinct_faces[n] = face

        # find the 3 norms which are perpendicular to each other
        keys_to_remove = []
        for key, value in self.distinct_norms.items():
            for key2, value2 in self.distinct_norms.items():
                if key == key2:
                    continue
                n1 = key.plane_normals / np.linalg.norm(key.plane_normals)
                n2 = key2.plane_normals / np.linalg.norm(key2.plane_normals)
                dot_product = np.dot(n1, n2)
                if (dot_product > 0.98 or dot_product < -0.98) and key2 not in keys_to_remove:
                    keys_to_remove.append(key)
                    break
        for key in keys_to_remove:
            self.distinct_norms.pop(key)
            self.distinct_faces.pop(key)

        assert len(self.distinct_norms) <= 3, "The number of distinct norms should be 3"
        return faces

    def optimize_orientation(self):
        planes_normals = [key.plane_normals for key, value in self.distinct_norms.items()]
        rotation_matrix = optimization.optimize_cubes_orientations(cube_normals=list(self.distinct_norms.values()),
                                                                   plane_normals=planes_normals)
        # rotate the cube
        self.cube_triangle_meshes.rotate(rotation_matrix)

    def optimize_size(self):
        cube_corners = []
        nodes = list(self.distinct_faces.keys())
        target_corners = []
        for i in nodes:
            target_corners.append(get_bb_edges_from_points(i))
        for i in self.distinct_faces.values():
            res = []
            for j in i:
                res.append(self.cube_triangle_meshes.vertices[j])
            cube_corners.append(res)

        cube_corners = np.array(cube_corners)
        target_corners = np.array(target_corners)
        e1e3_e2e4 = optimization.optimize_cubes_size(x_planes=cube_corners, y_planes=target_corners)


def get_bb_edges_from_points(node):
    points = node.projected_points
    pcd_point_original = o3d.geometry.PointCloud()
    pcd_point_original.points = o3d.utility.Vector3dVector(points)
    rpcd, R = rotate_point_cloud_align_z_axis(pcd_point_original, plane_normal=node.plane_normals)
    points = np.asarray(rpcd.points)
    #
    # # get the min and max points
    min_x = np.min(points[:, 0])
    max_x = np.max(points[:, 0])
    min_y = np.min(points[:, 1])
    max_y = np.max(points[:, 1])
    z = points[0, 2]

    # get the corner points
    corner_points = []
    corner_points.append([min_x, min_y, z])
    corner_points.append([max_x, min_y, z])
    corner_points.append([min_x, max_y, z])
    corner_points.append([max_x, max_y, z])

    # Rotate the corner points back to the original coordinate frame
    corner_points = np.linalg.inv(R) @ corner_points.T
    corner_points = corner_points.T
    # pcd2 = o3d.geometry.PointCloud()
    # pcd2.points = o3d.utility.Vector3dVector(corner_points)
    # pcd2.paint_uniform_color([0, 1, 1])
    #
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # vis.add_geometry(copy_pcd)
    # vis.add_geometry(rpcd)
    # vis.add_geometry(pcd2)
    # opt = vis.get_render_option()
    # opt.show_coordinate_frame = True
    # vis.run()
    return corner_points
