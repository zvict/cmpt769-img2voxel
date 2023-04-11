import copy

import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
import trimesh
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from main import rotate_point_cloud_align_z_axis
from tools.RGB_img_2_planes import get_all_planes


# in this file, we have a graph class, which is a list of nodes
# and a node class. For each node, we have a list of connected nodes

def do_lines_intersect(line1_start, line1_end, line2_start, line2_end):
    # Calculate slopes and y-intercepts of each line
    m1 = (line1_end[1] - line1_start[1]) / (line1_end[0] - line1_start[0])
    b1 = line1_start[1] - m1 * line1_start[0]
    m2 = (line2_end[1] - line2_start[1]) / (line2_end[0] - line2_start[0])
    b2 = line2_start[1] - m2 * line2_start[0]

    # Check if lines are parallel or coincident
    if m1 == m2:
        return b1 == b2

    # Calculate intersection point
    x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1

    # Check if intersection point is within range of both lines
    return (min(line1_start[0], line1_end[0]) <= x <= max(line1_start[0], line1_end[0])
            and min(line1_start[1], line1_end[1]) <= y <= max(line1_start[1], line1_end[1])
            and min(line2_start[0], line2_end[0]) <= x <= max(line2_start[0], line2_end[0])
            and min(line2_start[1], line2_end[1]) <= y <= max(line2_start[1], line2_end[1]))


class Node:
    def __init__(self, points, normals, labels, colors, plane_points, plane_normals, projected_points, raw_cube, id):
        self.connected_nodes = None
        self.points = points
        self.norm = normals
        self.labels = labels
        self.colors = colors
        self.plane_points = plane_points
        self.plane_normals = plane_normals
        self.projected_points = projected_points
        self.raw_cube = raw_cube
        self.id = id

    def add_connection(self, node):
        if self.connected_nodes is None:
            self.connected_nodes = []
        self.connected_nodes.append(node)

    def get_connections(self):
        return self.connected_nodes

    def get_norm(self):
        return self.norm

    def __print__(self):
        print(self.id)


class Graph:
    def __init__(self):
        self.nodes = {}
        self.pcd = o3d.geometry.PointCloud()
        self.line_set = None

    def add_node(self, node, id):
        self.nodes[id] = node
        new_points = np.concatenate((node.projected_points, np.asarray(self.pcd.points)), axis=0)
        self.pcd.points = o3d.utility.Vector3dVector(new_points)

    def get_nodes(self):
        return self.nodes

    def get_node(self, index):
        return self.nodes[index]

    def get_num_nodes(self):
        return len(self.nodes)

    def get_node_index(self, node):
        return self.nodes.index(node)

    def get_node_norm(self, node):
        return node.get_norm()

    def get_node_connections(self, node):
        return node.get_connections()

    def get_node_points(self, node):
        return node.points

    def get_node_plane_coord(self, node):
        return node.plane_coord

    def get_node_connected_nodes(self, node):
        return node.connected_nodes

    def find_all_connectives(self, point_type='projected', threshold=0.1):
        for n in self.nodes.values():
            self.find_connectivity(n.id, point_type=point_type, threshold=threshold)

    def get_intersection_two_planes_after_projection(self, node_1, node_2):
        # we first calculate the cross product of the two planes' normals
        # then we find the rotation matrix that rotates the cross product to the z axis
        # then we rotate the two planes to the x-y plane
        # then we find the intersection of the two planes in the x-y plane
        cross_product = np.cross(node_1.plane_normals, node_2.plane_normals)

        # we need to first Compute the angle between the normal vector and the z-axis (0, 0, 1)
        angle = np.arccos(np.dot(cross_product, np.array([0, 0, 1])))

        # finding the rotation axis
        axis = np.cross(cross_product, np.array([0, 0, 1]))
        axis = axis / np.linalg.norm(axis)

        R = np.array([[np.cos(angle) + axis[0] ** 2 * (1 - np.cos(angle)), axis[0] * axis[1] * (1 - np.cos(angle)) - axis[2] * np.sin(angle), axis[0] * axis[2] * (1 - np.cos(angle)) + axis[1] * np.sin(angle)],
                      [axis[1] * axis[0] * (1 - np.cos(angle)) + axis[2] * np.sin(angle), np.cos(angle) + axis[1] ** 2 * (1 - np.cos(angle)), axis[1] * axis[2] * (1 - np.cos(angle)) - axis[0] * np.sin(angle)],
                      [axis[2] * axis[0] * (1 - np.cos(angle)) - axis[1] * np.sin(angle), axis[2] * axis[1] * (1 - np.cos(angle)) + axis[0] * np.sin(angle), np.cos(angle) + axis[2] ** 2 * (1 - np.cos(angle))]])

        # visualization
        pcd_1 = o3d.geometry.PointCloud()
        pcd_1.points = o3d.utility.Vector3dVector(node_1.projected_points)
        pcd_1.paint_uniform_color([1, 0, 0])

        pcd_2 = o3d.geometry.PointCloud()
        pcd_2.points = o3d.utility.Vector3dVector(node_2.projected_points)
        pcd_2.paint_uniform_color([0, 1, 0])

        # we rotate the two planes to the x-y plane
        rotated_plane_1 = np.matmul(R, node_1.projected_points.T).T
        rotated_plane_2 = np.matmul(R, node_2.projected_points.T).T

        pcd_1_rotated = o3d.geometry.PointCloud()
        pcd_1_rotated.points = o3d.utility.Vector3dVector(rotated_plane_1)
        pcd_1_rotated.paint_uniform_color([1, 0, 0])

        pcd_2_rotated = o3d.geometry.PointCloud()
        pcd_2_rotated.points = o3d.utility.Vector3dVector(rotated_plane_2)
        pcd_2_rotated.paint_uniform_color([0, 1, 0])

        # finding the new normals
        new_normal_1 = np.matmul(R, node_1.plane_normals)
        new_normal_2 = np.matmul(R, node_2.plane_normals)

        # we will find the center point of the two rotated planes
        center_1 = np.mean(rotated_plane_1, axis=0)
        center_2 = np.mean(rotated_plane_2, axis=0)

        infinity_1 = center_1 + new_normal_1 * 10
        infinity_2 = center_2 + new_normal_2 * 10

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector([center_1, infinity_1, center_2, infinity_2])
        line_set.lines = o3d.utility.Vector2iVector([[0, 1], [2, 3]])
        line_set.colors = o3d.utility.Vector3dVector([[0, 0, 1], [0, 0, 1]])

        # center_1 and infinity_1 create a line(l1). center_2 and infinity_2 create a line (l2)
        # we want to see if l1 and l2 intersect
        intersection = do_lines_intersect(center_1[:2], infinity_1[:2], center_2[:2], infinity_2[:2])
        # if intersection:
        #     o3d.visualization.draw_geometries([pcd_1, pcd_2, pcd_1_rotated, pcd_2_rotated, line_set])
        return intersection

    def find_connectivity(self, point_id, point_type='projected', threshold=0.1):
        # in this function, we are going to find the connectivity of a node
        for i in range(len(self.nodes)):
            if i == point_id:
                continue
            if point_type == 'projected':
                if check_pc_intersects(self.nodes[i].projected_points,
                                       self.nodes[point_id].projected_points,
                                       thr=threshold):
                    self.nodes[i].add_connection(self.nodes[point_id])
                    self.nodes[point_id].add_connection(self.nodes[i])
            elif point_type == 'raw':
                if check_pc_intersects(self.nodes[i].plane_points,
                                       self.nodes[point_id].plane_points,
                                       thr=threshold):
                    self.nodes[i].add_connection(self.nodes[point_id])
                    self.nodes[point_id].add_connection(self.nodes[i])

    def plot_all_connectivity(self):
        # in this function, we are going to plot the connectivity of the nodes.
        # we have self.pcd, which is the point cloud of all the nodes
        # we loop over all nodes. for each node, we represent it by the center of it's projected points
        # and we loop over all of it's connections. for each connection, we represent it by the center of it's projected points
        # we draw a line between the two centers
        if self.line_set is None:
            points = [np.mean(np.asarray(self.nodes[i].projected_points), axis=0) for i in range(len(self.nodes))]
            lines = [[i, n.id] for i in range(len(self.nodes)) for n in self.nodes[i].connected_nodes]
            colors = [[1, 0, 0] for i in range(len(lines))]
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(points)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(colors)
            self.line_set = line_set
        o3d.visualization.draw_geometries([self.line_set, self.pcd])

    def plot_a_node_connectivity(self, id):
        origin_pcd = o3d.geometry.PointCloud()
        origin_pcd.points = o3d.utility.Vector3dVector(self.nodes[id].projected_points)
        origin_pcd.paint_uniform_color([0, 1, 0])

        destination_pcd = o3d.geometry.PointCloud()

        # check if the node connected_nodes is empty, find the connectivity
        if self.nodes[id].connected_nodes is None:
            self.find_connectivity(id, point_type='projected', threshold=0.1)
            self.prune_a_node_connectivity(node_id=id)

        # add the points of the connected nodes
        for node in self.nodes[id].connected_nodes:
            destination_pcd.points = o3d.utility.Vector3dVector(np.concatenate((np.asarray(destination_pcd.points), node.projected_points), axis=0))
        # draw the connectivity lines
        points = [np.mean(np.asarray(self.nodes[id].projected_points), axis=0)]
        lines = []
        # add the points of the connected nodes
        for idx, node in enumerate(self.nodes[id].connected_nodes):
            points.append(np.mean(np.asarray(node.projected_points), axis=0))
            lines.append([0, idx + 1])
        # add a point along the normal of the node
        normal_index = len(points)
        points.append(np.mean(np.asarray(self.nodes[id].projected_points), axis=0) + self.nodes[id].plane_normals * 0.5)
        for node in self.nodes[id].connected_nodes:
            points.append(np.mean(np.asarray(node.projected_points), axis=0) + node.plane_normals * 0.5)

        # add the normal lines
        for i in range(0, normal_index):
            lines.append([i, i + normal_index])

        colors = [[1, 0, 0] for i in range(normal_index)]

        for i in range(normal_index, len(points)):
            colors.append([0, 1, 0])

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        # set the color of pcd points to blue
        destination_pcd.paint_uniform_color([0, 0, 1])
        # get a copy of self.pcd and remove the _pcd points from it
        o3d.visualization.draw_geometries([self.pcd, line_set, origin_pcd, destination_pcd])

    def prune_a_node_connectivity(self, node_id, threshold=0.1):
        # in this function, we are looking for the nodes that
        # can build a cube. The filter to do it is that we calculate the angle between the
        # normals of the planes. If the angle is 90 degrees, then we can build a cube. Since the normals are
        # pointing outwards, we need to check the angle between the normals of the planes. If the angle is 90 degrees,
        # then we can build a cube. We have a noise, we need to consider a threshold. If the angle is between 80 and 100
        if self.nodes[node_id].connected_nodes is None:
            return None
        for connection in self.nodes[node_id].connected_nodes:
            angle = np.arccos(np.dot(self.nodes[node_id].plane_normals, connection.plane_normals))
            if angle < (np.pi / 2 - threshold) or angle > (np.pi / 2 + threshold):  # between 80 and 110 degrees
                print("Node: ", node_id, "Connection: ", connection.id, "Angle: ", angle)
            else:
                intersection = self.get_intersection_two_planes_after_projection(self.nodes[node_id], connection)
                if not intersection:
                    self.nodes[node_id].connected_nodes.remove(connection)
                    connection.connected_nodes.remove(self.nodes[node_id])

    def prune_all_connectivity(self, threshold=0.1):
        for node in self.nodes.values():
            self.prune_a_node_connectivity(node.id, threshold=threshold)


def graph_builder(clusters):
    """
    :param clusters: clusters is a dictionary containing the information of each cluster. The keys
    are "points", "normals", "labels", "colors", "plane_points", "plane_normals", "projected_points".
    We create a graph object and the nodes would be each item in the clusters dictionary.
    :return: a graph object
    """
    graph = Graph()
    for i in range(len(clusters["points"])):
        node = Node(points=clusters["points"][i],
                    normals=clusters["normals"][i],
                    labels=clusters["labels"][i],
                    colors=clusters["colors"][i],
                    plane_points=clusters["plane_points"][i],
                    plane_normals=clusters["plane_normals"][i],
                    projected_points=clusters["projected_points"][i],
                    raw_cube=clusters["raw_cubes"][i],
                    id=i)
        graph.add_node(node, i)
    # set the color of the graph.pcd to red
    graph.pcd.paint_uniform_color([1, 0, 0])
    return graph


def get_plane_corners(plane_points, plane_normal):
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(plane_points)

    # rotate the plane to the xy plane
    r_pcl, R = rotate_point_cloud_align_z_axis(pcl, plane_normal=plane_normal)
    plane_points_r = np.asarray(r_pcl.points)
    min_x = np.min(plane_points_r[:, 0])
    max_x = np.max(plane_points_r[:, 0])
    min_y = np.min(plane_points_r[:, 1])
    max_y = np.max(plane_points_r[:, 1])

    corners = np.zeros((4, 3))
    corners[0, :] = [min_x, min_y, plane_points_r[0, 2]]
    corners[1, :] = [max_x, min_y, plane_points_r[0, 2]]
    corners[2, :] = [max_x, max_y, plane_points_r[0, 2]]
    corners[3, :] = [min_x, max_y, plane_points_r[0, 2]]

    # rotate the corners back to the original plane using the rotation matrix
    corner_original = np.linalg.inv(R) @ corners.T
    plane_points_r = np.linalg.inv(R) @ plane_points_r.T

    # plane points are nx3, however, the corners are 3xn
    # we need to transpose the corners to make them nx3
    corner_original = corner_original.T
    plane_points_r = plane_points_r.T

    assert np.allclose(plane_points_r[0:2, :], plane_points[0:2, :])

    return corner_original


# def create_tensor_cube_mesh(corners): # create a tensor mesh cube from the corners
#     triangles = np.array([[0, 1, 2], [2, 3, 0], [0, 4, 5], [5, 1, 0], [1, 5, 6], [6, 2, 1], [2, 6, 7], [7, 3, 2], [3, 7, 4], [4, 0, 3], [5, 4, 7], [7, 6, 5]])
#     device = o3d.core.Device("CPU:0")
#     dtype_f = o3d.core.float32
#     dtype_i = o3d.core.int32

#     mesh = o3d.t.geometry.TriangleMesh()
#     mesh.vertex.positions = o3d.core.Tensor(corners, device=device, dtype=dtype_f)
#     mesh.triangle.indices = o3d.core.Tensor(triangles, device=device, dtype=dtype_i)
#     mesh.compute_vertex_normals()
#     mesh.compute_triangle_normals()

#     return mesh


def tmesh2mesh(tmesh):
    mesh = o3d.geometry.TriangleMesh()
    mesh.triangles = o3d.utility.Vector3iVector(tmesh.triangle.indices.numpy())
    mesh.vertices = o3d.utility.Vector3dVector(tmesh.vertex.positions.numpy())
    return mesh


def mesh2tmesh(mesh):
    tmesh = o3d.t.geometry.TriangleMesh()
    tmesh.vertex.positions = o3d.core.Tensor(np.asarray(mesh.vertices), dtype=o3d.core.float32)
    tmesh.triangle.indices = o3d.core.Tensor(np.asarray(mesh.triangles), dtype=o3d.core.int32)
    return tmesh


def mesh2trimesh(mesh):
    tmesh = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.triangles), process=False)
    return tmesh


def trimesh2mesh(trimesh):
    mesh = o3d.geometry.TriangleMesh()
    mesh.triangles = o3d.utility.Vector3iVector(trimesh.faces)
    mesh.vertices = o3d.utility.Vector3dVector(trimesh.vertices)
    return mesh


def get_cube_intersection(cube1, cube2, method='trimesh'):
    if method == 'trimesh':  # trimesh has less outliers then o3d, but needs Blender
        trimesh1 = mesh2trimesh(o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(cube1.get_minimal_oriented_bounding_box()))
        trimesh2 = mesh2trimesh(o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(cube2.get_minimal_oriented_bounding_box()))
        intersection = trimesh.boolean.intersection([trimesh1, trimesh2])
        if isinstance(intersection, trimesh.Scene):
            return None
        return trimesh2mesh(intersection)

    elif method == 'o3d':
        tmesh1 = mesh2tmesh(o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(cube1.get_minimal_oriented_bounding_box()))
        tmesh2 = mesh2tmesh(o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(cube2.get_minimal_oriented_bounding_box()))
        intersection = tmesh1.boolean_intersection(tmesh2)
        # intersection = tmesh1.boolean_difference(tmesh2)
        # intersection = tmesh1.boolean_union(tmesh2)
        if intersection.vertex.positions.shape[0] == 0:
            return None
        return tmesh2mesh(intersection)


def create_cuboid(corners):
    # Define the edges of the cuboid
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [0, 4], [1, 5], [2, 6], [3, 7], [4, 5], [5, 6], [6, 7], [7, 4]])

    # Create a mesh from the corners and edges
    mesh = o3d.geometry.LineSet()
    mesh.points = o3d.utility.Vector3dVector(corners)
    mesh.lines = o3d.utility.Vector2iVector(edges)
    mesh.colors = o3d.utility.Vector3dVector(np.array([[1, 0, 0] for i in range(len(edges))]))

    return mesh


def check_pc_intersects(pc1, pc2, thr=0.01):
    target_pc = pc1 if pc1.shape[0] > pc2.shape[0] else pc2
    query_pc = pc2 if pc1.shape[0] > pc2.shape[0] else pc1
    tree = KDTree(target_pc)
    dist, _ = tree.query(query_pc)
    return np.any(dist <= thr)


def get_raw_cubes(graph):
    graph['raw_cubes'] = []
    expansion = graph['length_max'] * 3
    for i in range(len(graph['projected_points'])):
        # our goal is to create a cube for each plane.
        # we need to find the bounding box of the plane.
        # Then, we need to expand the bounding box by a certain amount along the normal direction.
        # Then, we need to create a cube with the expanded bounding box.
        # We need to do this for each plane.

        # 1: find the bounding box of the plane
        # 2: get the 4 corners of the bounding box
        # 3: calculate the next 4 corners by expanding the previous 4 corners along the normal direction
        # 4: create a cube with the 8 corners

        # 1 and 2: find the bounding box of the plane
        corners = get_plane_corners(graph['projected_points'][i], graph['plane_normals'][i])

        # 3: calculate the next 4 corners by expanding the previous 4 corners along the normal direction
        new_corners = np.zeros((8, 3))
        new_corners[0:4, :] = corners

        # expand the corners along the normal direction
        new_corners[4:8, :] = corners - expansion * graph['plane_normals'][i]

        # create a cube with the 8 corners
        cube = create_cuboid(new_corners)
        graph['raw_cubes'].append(cube)
        # o3d.visualization.draw_geometries([cube])
        # break

    # visualize the raw cubes
    # o3d.visualization.draw_geometries(graph['raw_cubes'])


def longest_common_subsequence(list1, list2):
    list1.sort()
    list2.sort()
    m, n = len(list1), len(list2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if list1[i - 1] == list2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    subsequence = []
    i, j = m, n
    while i > 0 and j > 0:
        if list1[i - 1] == list2[j - 1]:
            subsequence.append(list1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    subsequence.reverse()
    return subsequence


def longest_common_subsequence_multiple(lists):
    if not lists:
        return []
    if len(lists) == 1:
        return lists[0]

    result = lists[0]
    for i in range(1, len(lists)):
        result = longest_common_subsequence(result, lists[i])

    return result


def get_final_cubes(graph):
    pass


def main():
    clusters = get_all_planes()
    get_raw_cubes(clusters)

    # cube_id1 = 0
    # cube_id2 = 14
    # # if using method='o3d' to find the intersection, cube_id1 = 0, cube_id2 = 4 will fail
    # intersection_mesh = get_cube_intersection(clusters['raw_cubes'][cube_id1], clusters['raw_cubes'][cube_id2], method='trimesh')
    #
    # intersection = []
    # if intersection_mesh is not None:
    #     # intersection_mesh = tmesh2mesh(intersection_tmesh)
    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(intersection_mesh.vertices)
    #     # pcd = intersection_mesh.sample_points_uniformly(number_of_points=1000)
    #     bbox = pcd.get_minimal_oriented_bounding_box()
    #     # bbox = intersection_mesh.get_minimal_oriented_bounding_box()    # get bounding box directly from the mesh does not work
    #     # bbox = intersection_mesh.get_oriented_bounding_box()
    #     # bbox = intersection_mesh.get_axis_aligned_bounding_box()
    #     bbox.color = [0, 1, 0]
    #     intersection.append(intersection_mesh)
    #     intersection.append(pcd)
    #     intersection.append(bbox)
    #
    # o3d.visualization.draw_geometries([clusters['raw_cubes'][cube_id1], clusters['raw_cubes'][cube_id2]] + intersection)

    # lists = [[1, 2, 5, 12, 7, 3, 6, 9, 23], [0, 9, 6, 4, 5, 23], [4, 5, 8, 23, 3, 6, 1, 16, 9, 45]]
    # result = longest_common_subsequence_multiple(lists)
    # print(result)

    # build the graph
    graph = graph_builder(clusters)
    graph.find_all_connectives()

    # prune connections
    # print("*" * 50)
    graph.prune_all_connectivity()

    # # visualize the graph
    # nodes = graph.get_nodes().values()
    #
    # for n in nodes:
    #     graph.plot_a_node_connectivity(n.id)

    # get final cubes
    get_final_cubes(graph)

    # x = 0


if __name__ == "__main__":
    main()
