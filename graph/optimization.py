import numpy as np
from scipy.optimize import minimize


def optimize_cubes_orientations(cube_normals, plane_normals):
    """Optimize the orientation of the cubes in the list.

    Parameters
    ----------
    cube_normals : list numpy.ndarray of shape (3,)
        List of cubes' normals.
    plane_normals : list numpy.ndarray of shape (3,)
        List of planes' normals.
    We try to find the best rotation matrix R that minimize the sum of the cosine of the angle between
    the cube normals and the plane normals. The cosine of the angle between two vectors is the dot product.
    """
    cube_normals = np.array(cube_normals)
    plane_normals = np.array(plane_normals)

    # normalize the normals
    cube_normals = cube_normals / np.linalg.norm(cube_normals, axis=1)[:, np.newaxis]
    plane_normals = plane_normals / np.linalg.norm(plane_normals, axis=1)[:, np.newaxis]

    def cost_function(theta):
        R = rotation_matrix(theta)
        rotated_x = np.dot(R, cube_normals.T).T
        return np.sum(np.abs(rotated_x - plane_normals))

    def rotation_matrix(theta):
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        Rx = np.array([[1, 0, 0], [0, cos_theta[0], -sin_theta[0]], [0, sin_theta[0], cos_theta[0]]])
        Ry = np.array([[cos_theta[1], 0, sin_theta[1]], [0, 1, 0], [-sin_theta[1], 0, cos_theta[1]]])
        Rz = np.array([[cos_theta[2], -sin_theta[2], 0], [sin_theta[2], cos_theta[2], 0], [0, 0, 1]])
        return np.dot(Rz, np.dot(Ry, Rx))

    theta0 = np.zeros(3)
    result = minimize(cost_function, theta0)

    return rotation_matrix(result.x)


def optimize_cubes_size(cubes_edge_pairs, planes_areas):
    """Optimize the size of the cubes in the list.

    Parameters
    ----------
    cubes_edge_pairs : list numpy.ndarray of shape (2, 3)
        List of cubes' edge pairs.
    planes_areas : list float
        List of planes' areas.
    We try to find the best size of the cubes that minimize the sum of the difference between the cubes' areas
    and the planes' areas.
    """
    cubes_edge_pairs = np.array(cubes_edge_pairs)
    planes_areas = np.array(planes_areas)

    def cost_function(x):
        return np.sum(np.abs(x[0] * x[1] - planes_areas))

    result = minimize(cost_function, cubes_edge_pairs)

    return result.x
