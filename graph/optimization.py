import numpy as np
from scipy.optimize import minimize


# def optimize_cubes_orientations(cube_normals, plane_normals):
#     """Optimize the orientation of the cubes in the list.
#
#     Parameters
#     ----------
#     cube_normals : list numpy.ndarray of shape (3,)
#         List of cubes' normals.
#     plane_normals : list numpy.ndarray of shape (3,)
#         List of planes' normals.
#     We try to find the best rotation matrix R that minimize the sum of the cosine of the angle between
#     the cube normals and the plane normals. The cosine of the angle between two vectors is the dot product.
#     """
#     cube_normals = np.array(cube_normals)
#     plane_normals = np.array(plane_normals)
#
#     def cost_function(rotation_matrix):
#         rotation_matrix = rotation_matrix.reshape(3, 3)
#         rotated_normals = cube_normals.dot(rotation_matrix)
#         errors = []
#         for rotated_normal, plane_normal in zip(rotated_normals, plane_normals):
#             dot_product = np.dot(rotated_normal, plane_normal)
#             errors.append(1 - dot_product)
#         return sum(errors)
#
#     rotation_matrix = np.eye(3)
#
#     # We use the L-BFGS-B algorithm to find the best rotation matrix
#     # res = minimize(cost_function, initial_rotation_matrix, method='BFGS')
#     # this time we use gradient descent
#     learning_rate = 0.1
#     num_iterations = 1000
#
#     def cost_gradient(rotation_matrix):
#         rotated_normals = cube_normals.dot(rotation_matrix)
#         gradients = np.zeros((3, 3))
#         for plane_normal in plane_normals:
#             dot_products = rotated_normals.dot(plane_normal)
#             max_dot_index = np.argmax(dot_products)
#             if dot_products[max_dot_index] < 1:
#                 continue
#             max_dot_normal = rotated_normals[max_dot_index]
#             max_dot_error = 2 * (dot_products[max_dot_index] - 1)
#             max_dot_gradient = np.outer(max_dot_normal, plane_normal)
#             gradients += max_dot_error * max_dot_gradient
#         return gradients
#
#     # Perform gradient descent
#     for i in range(num_iterations):
#         gradient = cost_gradient(rotation_matrix)
#         rotation_matrix -= learning_rate * gradient
#         cost = cost_function(rotation_matrix)
#         print("Iteration {}: cost = {}".format(i, cost))
#
#     return rotation_matrix

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


def test_optimization():
    # cube_normals = [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]
    cube_normals = [[0.0, 0.3, 0.9]]
    # plane_normals = [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]
    plane_normals = [[0.3, 0.5, 0.0]]
    rotation_matrix = optimize_cubes_orientations(cube_normals=cube_normals, plane_normals=plane_normals)
    print(rotation_matrix)
