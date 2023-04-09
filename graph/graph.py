import copy

import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
from main import rotate_point_cloud_align_z_axis
from tools.RGB_img_2_planes import get_all_planes


# in this file, we have a graph class, which is a list of nodes
# and a node class. For each node, we have a list of connected nodes


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
        # add the points of the connected nodes
        for node in self.nodes[id].connected_nodes:
            points.append(np.mean(np.asarray(node.projected_points), axis=0))
        lines = [[id, n.id] for n in self.nodes[id].connected_nodes]
        colors = [[1, 0, 0] for i in range(len(lines))]
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
            if angle > (np.pi / 2 - threshold) and angle < (np.pi / 2 + threshold):
                print("Node: ", node_id, "Connection: ", connection.id, "Angle: ", angle)
            else:
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
        new_corners[4:8, :] = corners + expansion * graph['plane_normals'][i]

        # create a cube with the 8 corners
        cube = create_cuboid(new_corners)
        graph['raw_cubes'].append(cube)

    # visualize the raw cubes
    o3d.visualization.draw_geometries(graph['raw_cubes'])


def main():
    # get the clusters and the raw cubes
    clusters = get_all_planes()
    get_raw_cubes(clusters)

    # build the graph
    graph = graph_builder(clusters)
    graph.find_all_connectives()

    # prune connections
    print("*" * 50)
    graph.prune_all_connectivity()

    # visualize the graph
    nodes = graph.get_nodes().values()

    for n in nodes:
        graph.plot_a_node_connectivity(n.id)

    x = 0


if __name__ == "__main__":
    main()
