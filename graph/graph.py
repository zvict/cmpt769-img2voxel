# in this file, we have a graph class, which is a list of nodes
# and a node class. For each node, we have a list of connected nodes


class Node:
    def __init__(self, norm, plane_coord, points):
        """
        :param norm: normal vector of the plane
        :param plane_coord: coordinates of the plane
        :param points: points of the plane
        """
        self.connected_nodes = []
        self.norm = norm
        self.plane_coord = plane_coord
        self.points = points

    def add_connection(self, node):
        self.connected_nodes.append(node)

    def get_connections(self):
        return self.connected_nodes

    def get_norm(self):
        return self.norm


class Graph:
    def __init__(self):
        self.nodes = []

    def add_node(self, node):
        self.nodes.append(node)

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


def graph_builder():