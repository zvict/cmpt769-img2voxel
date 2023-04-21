import open3d as o3d


def get_triangle_mesh_from_oriented_bounding_box(aabb):
    corners = aabb.get_box_points()
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(corners)
    faces = [
        [0, 1, 2], [0, 2, 3],
        [0, 3, 7], [0, 7, 4],
        [1, 5, 6], [1, 6, 2],
        [2, 6, 7], [2, 7, 3],
        [0, 4, 5], [0, 5, 1],
        [4, 7, 6], [4, 6, 5],
    ]

    # Add the faces to the TriangleMesh object
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    return mesh


def get_triangle_mesh_from_aabb(aabb):
    vertices = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                vertex = [aabb.min_bound[0] + i * aabb.get_extent()[0],
                          aabb.min_bound[1] + j * aabb.get_extent()[1],
                          aabb.min_bound[2] + k * aabb.get_extent()[2]]
                vertices.append(vertex)
    faces = [[0, 1, 3], [0, 3, 2], [0, 2, 6], [0, 6, 4], [0, 4, 5], [0, 5, 1], [7, 6, 2], [7, 2, 3], [7, 3, 1], [7, 1, 5], [7, 5, 4], [7, 4, 6]]
    cube_mesh = o3d.geometry.TriangleMesh()
    cube_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    cube_mesh.triangles = o3d.utility.Vector3iVector(faces)
    return cube_mesh


def save_triangle_mesh(triangle_mesh, file_name):
    o3d.io.write_triangle_mesh(file_name, triangle_mesh)


def load_triangle_mesh(file_name):
    return o3d.io.read_triangle_mesh(file_name)
