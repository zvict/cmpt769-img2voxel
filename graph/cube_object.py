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
        # visualize the corners
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(corners[:, 0], corners[:, 1], corners[:, 2])
        # for i in range(8):
        #     ax.text(corners[i, 0], corners[i, 1], corners[i, 2], "({}, {}, {})".format(corners[i, 0], corners[i, 1], corners[i, 2]))
        # plt.show()

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
        d0 = [[0, 4], [2, 6], [1, 5], [3, 7]]
        d1 = [[0, 2], [4, 6], [1, 3], [5, 7]]
        d2 = [[0, 1], [2, 3], [4, 5], [6, 7]]
        input_dict = {}  # keys are d_id_j, values are the corresponding edges length
        target_dict = {}  # keys are d_id_j, values are the corresponding edges length
        # we have to create a dictionary of new edges
        for node, corners in self.distinct_faces.items():
            edge_1 = ""
            edge_2 = ""
            idx_1 = 0
            idx_2 = 0
            if [corners[0], corners[1]] in d0:
                edge_1 = "d0"
                idx_1 = 0
            elif [corners[0], corners[1]] in d1:
                edge_1 = "d1"
                idx_1 = 1
            elif [corners[0], corners[1]] in d2:
                edge_1 = "d2"
                idx_1 = 2
            if [corners[0], corners[3]] in d0:
                edge_2 = "d0"
                idx_2 = 0
            elif [corners[0], corners[3]] in d1:
                edge_2 = "d1"
                idx_2 = 1
            elif [corners[0], corners[3]] in d2:
                edge_2 = "d2"
                idx_2 = 2

            c = get_bb_edges_from_points(node)
            if idx_1 < idx_2:
                key = edge_1 + edge_2
                e1 = np.linalg.norm(self.cube_triangle_meshes.vertices[corners[0]] - self.cube_triangle_meshes.vertices[corners[1]])
                e2 = np.linalg.norm(self.cube_triangle_meshes.vertices[corners[0]] - self.cube_triangle_meshes.vertices[corners[3]])
                e1_target = np.linalg.norm(c[0] - c[1])
                e2_target = np.linalg.norm(c[0] - c[3])
            else:
                key = edge_2 + edge_1
                e2 = np.linalg.norm(self.cube_triangle_meshes.vertices[corners[0]] - self.cube_triangle_meshes.vertices[corners[1]])
                e1 = np.linalg.norm(self.cube_triangle_meshes.vertices[corners[0]] - self.cube_triangle_meshes.vertices[corners[3]])
                e2_target = np.linalg.norm(c[0] - c[1])
                e1_target = np.linalg.norm(c[0] - c[3])
            if key not in input_dict:
                input_dict[key] = [[e1, e2]]
                target_dict[key] = [[e1_target, e2_target]]
            else:
                input_dict[key].append([e1, e2])
                target_dict[key].append([e1_target, e2_target])

        output_dict = optimization.optimize_cubes_size(x_planes=input_dict, y_planes=target_dict)
        result = {}
        for key, value in output_dict.items():
            idx_1 = key[0: 2]
            idx_2 = key[2:]
            if idx_1 not in result:
                result[idx_1] = value[0]
            if idx_2 not in result:
                result[idx_2] = value[1]
        # scale the cube
        cdcdcv = 0


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
    corner_points = np.array(corner_points)

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
