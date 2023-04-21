import numpy as np
import open3d as o3d


class CustomCube():
    def __init__(self, cube_triangle_meshes, nodes):
        self.nodes = nodes
        self.cube_triangle_meshes = cube_triangle_meshes
        self.cube_norms = []
        self.distinct_norms = {}  # key: plane_normals, value: best aligned norm on cube
        self.faces = []
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
        # we will return the norms of all surfaces of the cube
        corners = np.asarray(self.cube_triangle_meshes.vertices)

        # Define the indices of each face of the cube
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
        return normals

    def get_nearest_aligned_norm(self, node):
        max_dot_product = 0
        best_norm = None
        for idx, norm in enumerate(self.cube_norms):
            n1 = norm / np.linalg.norm(norm)
            n2 = node.plane_normals / np.linalg.norm(node.plane_normals)
            dot_product = np.dot(n1, n2)
            if dot_product > max_dot_product:
                max_dot_product = dot_product
                best_norm = norm
        return best_norm

    def get_distinct_norms(self):
        # for each cube we consider only 3 norms. If the cube
        # has more than 3 norms, we select the 3 norms which are perpendicular to each other
        self.cube_norms = self.get_norms_from_cube_triangle_meshes()
        for n in self.nodes:
            nearest_norm = self.get_nearest_aligned_norm(n)
            self.distinct_norms[n] = nearest_norm
        if len(self.distinct_norms) <= 3:
            return

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

        assert len(self.distinct_norms) <= 3, "The number of distinct norms should be 3"
