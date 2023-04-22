import numpy as np
from matplotlib import pyplot as plt
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


def optimize_cubes_size(x_planes, y_planes):
    """Optimize the size of the cubes in the list.

    Parameters
    ----------
    x_planes: cubes_edge_pairs : list numpy.ndarray of shape (2, 3)
        List of cubes' edge pairs.
    y_planes: planes_areas : list float
        List of planes' areas.
    We try to find the best size of the cubes that minimize the sum of the difference between the cubes' areas
    and the planes' areas.
    """

    def run_optimize(delta, e1e3, e2e4, e1e3_gt, e2e4_gt):
        res = minimize(
            area_difference,
            delta,
            args=(e1e3, e2e4, e1e3_gt, e2e4_gt),
        )
        x_opt = res.x
        return x_opt

    def area_difference(delta, e1e3, e2e4, e1e3_gt, e2e4_gt):
        delta = delta.reshape(-1, 2)
        x_areas = (e1e3 + delta[:, 0]) * (e2e4 + delta[:, 1])
        y_areas = e1e3_gt * e2e4_gt
        return np.sum((x_areas - y_areas) ** 2)

    delta_init = np.ones((x_planes.shape[0], 2)) * 0.1  # (n, 2), n is the number of cubes, delta[:, 0] is e1e3, delta[:, 1] is e2e4
    _e1e3 = np.linalg.norm(x_planes[:, 0] - x_planes[:, 1], axis=1)  # 1 x n
    _e2e4 = np.linalg.norm(x_planes[:, 0] - x_planes[:, 2], axis=1)  # 1 x n
    _e1e3_gt = np.linalg.norm(y_planes[:, 0] - y_planes[:, 1], axis=1)
    _e2e4_gt = np.linalg.norm(y_planes[:, 0] - y_planes[:, 2], axis=1)

    for i in range(10):
        delta_optimized = run_optimize(delta_init, _e1e3, _e2e4, _e1e3_gt, _e2e4_gt).reshape(-1, 2)
        _e1e3 = _e1e3 + delta_optimized[:, 0]
        _e2e4 = _e2e4 + delta_optimized[:, 1]

    # x_e1 = np.linalg.norm(x_planes[:, 0] - x_planes[:, 1], axis=1).reshape(-1, 1)
    # x_e2 = np.linalg.norm(x_planes[:, 0] - x_planes[:, 2], axis=1).reshape(-1, 1)
    # x_u1 = (x_planes[:, 1] - x_planes[:, 0]) / x_e1
    # x_u2 = (x_planes[:, 2] - x_planes[:, 0]) / x_e2
    # _e1e3 = _e1e3.reshape(-1, 1)
    # _e2e4 = _e2e4.reshape(-1, 1)
    # x_planes[:, 1] = x_planes[:, 0] + _e1e3 * x_u1
    # x_planes[:, 2] = x_planes[:, 0] + _e2e4 * x_u2
    # x_planes[:, 3] = x_planes[:, 1] + x_planes[:, 2]

    e1e3_e2e4 = np.concatenate((_e1e3.reshape(-1, 1), _e2e4.reshape(-1, 1)), axis=1)

    return e1e3_e2e4  # n x 2!


def test_optimize_cube_size():
    # x_planes are some random rectangles
    x_planes = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], [[0, 0, 0], [6, 0, 0], [0, 6, 0], [3, 3, 0]]])
    y_planes = np.array([[[0, 0, 0], [2, 0, 0], [0, 2, 0], [4, 4, 0]], [[0, 0, 0], [4, 0, 0], [0, 4, 0], [8, 8, 0]]])

    print("*" * 80)
    x_planes_copy = np.copy(x_planes)
    x_planes_optimized = optimize_cubes_size(x_planes, y_planes)

    print("x_planes: " + str(x_planes_copy))
    print("y_planes: " + str(y_planes))
    print("x_planes_optimized: " + str(x_planes_optimized))

    print("initial area: " + str(np.abs(np.linalg.norm(x_planes_copy[:, 1] - x_planes_copy[:, 0], axis=1) * np.linalg.norm(x_planes_copy[:, 2] - x_planes_copy[:, 0], axis=1))))
    print("target area: " + str(np.abs(np.linalg.norm(y_planes[:, 1] - y_planes[:, 0], axis=1) * np.linalg.norm(y_planes[:, 2] - y_planes[:, 0], axis=1))))
    print("optimized area:" + str(np.abs(np.linalg.norm(x_planes_optimized[:, 1] - x_planes_optimized[:, 0], axis=1) * np.linalg.norm(x_planes_optimized[:, 2] - x_planes_optimized[:, 0], axis=1))))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x_planes_copy[:, :, 0], x_planes_copy[:, :, 1], x_planes_copy[:, :, 2], c='r')
    ax.scatter(y_planes[:, :, 0], y_planes[:, :, 1], y_planes[:, :, 2], c='b')
    ax.scatter(x_planes_optimized[:, :, 0], x_planes_optimized[:, :, 1], x_planes_optimized[:, :, 2], c='g')
    for i in range(4):
        ax.text(x_planes_copy[:, i, 0], x_planes_copy[:, i, 1], x_planes_copy[:, i, 2], f'({x_planes_copy[:, i, 0]}, {x_planes_copy[:, i, 1]}, {x_planes_copy[:, i, 2]})')
        ax.text(y_planes[:, i, 0], y_planes[:, i, 1], y_planes[:, i, 2], f'({y_planes[:, i, 0]}, {y_planes[:, i, 1]}, {y_planes[:, i, 2]})')
        ax.text(x_planes_optimized[:, i, 0], x_planes_optimized[:, i, 1], x_planes_optimized[:, i, 2], f'({x_planes_optimized[:, i, 0]}, {x_planes_optimized[:, i, 1]}, {x_planes_optimized[:, i, 2]})')
    plt.show()
