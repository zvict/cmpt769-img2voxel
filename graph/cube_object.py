import numpy as np
import open3d as o3d


class CustomCube():
    def __init__(self, cube_triangle_meshes, nodes):
        self.nodes = nodes
        self.cube_triangle_meshes = cube_triangle_meshes
        self.norms = []
        for n in nodes:
            self.norms.append(n.plane_normals)
        self.faces = []
        self.visualize_cube()
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
            normals.append(normal)
        return normals

    def get_distinct_norms(self):
        # for each cube we consider only 3 norms. If the cube
        # has more than 3 norms, we select the 3 norms which are perpendicular to each other
        normals = self.get_norms_from_cube_triangle_meshes()
        if len(self.norms) <= 3:
            return

        # find the 3 norms which are perpendicular to each other
        tmp_norms = []
