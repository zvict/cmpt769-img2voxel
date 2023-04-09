import matplotlib.pyplot as plt
import numpy as np
from skspatial.objects import Plane, Points


def point2plane_lstsq(points):
    xs, ys, zs = points[:, 0], points[:, 1], points[:, 2]
    # do fit
    tmp_A = []
    tmp_b = []
    for i in range(len(xs)):
        tmp_A.append([xs[i], ys[i], 1])
        tmp_b.append(zs[i])
    b = np.matrix(tmp_b).T
    A = np.matrix(tmp_A)

    # Manual solution
    fit = (A.T * A).I * A.T * b
    errors = b - A * fit
    residual = np.linalg.norm(errors)

    # Or use Scipy
    # from scipy.linalg import lstsq
    # fit, residual, rnk, s = lstsq(A, b)

    print("solution: %f x + %f y + %f = z" % (fit[0], fit[1], fit[2]))
    print("errors: \n", errors)
    print("residual:", residual)

    return fit, residual


def sks_point2plane(points):
    plane = Plane.best_fit(points)
    print("plane: ", plane)
    return plane


def project_points2plane(points, plane):
    # points: N x 3
    # plane: 3 x 1
    # return: N x 3
    N = points.shape[0]
    plane = plane.reshape((3, 1))
    plane = np.tile(plane, (1, N))
    points = points.T
    proj_points = points - np.multiply(plane, np.dot(plane.T, points))
    proj_points = proj_points.T
    return proj_points


# def sks_project_points2plane(points, plane):
#     proj_points = []
#     for point in points:
#         proj_points.append(plane.project_point(point))
#     proj_points = np.array(proj_points)
#     return proj_points


# def sympy_point2plane(points):
#     # points: N x 3
#     # return: 3 x 1
#     points = [sp_plane.Point3D(point[0], point[1], point[2]) for point in points]
#     plane = sp_plane.Plane(points)
#     print("plane: ", plane)
#     return plane


def project_points_on_plane(point_on_plane, plane_normal, points):
    """
    Given a point on a plane and the plane normal vector, project a list of points onto the plane.
    """
    # Calculate the projection of each point onto the plane
    # Projection formula: p' = p - ((p - p0) . n) * n, where p is the point, p0 is a point on the plane,
    # n is the normal vector of the plane, and . is the dot product.
    projected_points = points - np.dot(points - point_on_plane, plane_normal)[:, np.newaxis] * plane_normal

    return projected_points
