import cv2
import numpy as np
import os


K = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # depth camera's intrinsic


def get_surface_normal_by_depth(depth, K=None):
    """
    depth: (h, w) of float, the unit of depth is meter
    K: (3, 3) of float, the depth camere's intrinsic
    """
    K = [[1, 0], [0, 1]] if K is None else K
    fx, fy = K[0][0], K[1][1]

    dz_dv, dz_du = np.gradient(depth)  # u, v mean the pixel coordinate in the image
    # u*depth = fx*x + cx --> du/dx = fx / depth
    du_dx = fx / depth  # x is xyz of camera coordinate
    dv_dy = fy / depth

    dz_dx = dz_du * du_dx
    dz_dy = dz_dv * dv_dy
    # cross-product (1,0,dz_dx)X(0,1,dz_dy) = (-dz_dx, -dz_dy, 1)
    normal_cross = np.dstack((-dz_dx, -dz_dy, np.ones_like(depth)))
    # normalize to unit vector
    normal_unit = normal_cross / np.linalg.norm(normal_cross, axis=2, keepdims=True)
    # set default normal to [0, 0, 1]
    normal_unit[~np.isfinite(normal_unit).all(2)] = [0, 0, 1]
    return normal_unit


def get_normal_map_by_point_cloud(depth, K):
    height, width = depth.shape

    def normalization(data):
        mo_chang = np.sqrt(
            np.multiply(data[:, :, 0], data[:, :, 0])
            + np.multiply(data[:, :, 1], data[:, :, 1])
            + np.multiply(data[:, :, 2], data[:, :, 2])
        )
        mo_chang = np.dstack((mo_chang, mo_chang, mo_chang))
        return data / mo_chang

    x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x = x.reshape([-1])
    y = y.reshape([-1])
    xyz = np.vstack((x, y, np.ones_like(x)))
    pts_3d = np.dot(np.linalg.inv(K), xyz * depth.reshape([-1]))
    pts_3d_world = pts_3d.reshape((3, height, width))
    f = (
        pts_3d_world[:, 1 : height - 1, 2:width]
        - pts_3d_world[:, 1 : height - 1, 1 : width - 1]
    )
    t = (
        pts_3d_world[:, 2:height, 1 : width - 1]
        - pts_3d_world[:, 1 : height - 1, 1 : width - 1]
    )
    normal_map = np.cross(f, t, axisa=0, axisb=0)
    normal_map = normalization(normal_map)
    return normal_map, pts_3d_world


def normalize_depth(depth):
    return (depth - depth.min()) / (depth.max() - depth.min())


if __name__ == "__main__":
    img_dir = './data'
    img_id = '0022'
    # depth_dir = 'SI_R20_lowres'
    depth_dir = 'SI_R20'
    
    # depth = cv2.imread(os.path.join(img_dir, depth_dir, img_id + '_final_midas_v2_o2m_lowres.png'), cv2.IMREAD_ANYDEPTH)
    depth = cv2.imread(os.path.join(img_dir, depth_dir, img_id + '_final_midas_v2_o2m.png'), cv2.IMREAD_ANYDEPTH)
    depth = depth.astype(np.float32)
    depth = normalize_depth(depth)

    vis_normal = lambda normal: np.uint8((normal + 1) / 2 * 255)[..., ::-1]

    normal1 = get_surface_normal_by_depth(depth, K)    #  spend time: 60ms
    normal2, pcl = get_normal_map_by_point_cloud(depth, K)  #  spend time: 90ms

    cv2.imwrite("data/segmentations/normal1.png", vis_normal(normal1))
    cv2.imwrite("data/segmentations/normal2.png", vis_normal(normal2))
    # cv2.imwrite("normal1.png", (normal1))
    # cv2.imwrite("normal2.png", (normal2))

    normal1 = cv2.imread("data/segmentations/normal1.png")
    normal2 = cv2.imread("data/segmentations/normal2.png")

    print(normal1.shape, normal1.max(), normal1.min())
    print(normal2.shape, normal2.max(), normal2.min())