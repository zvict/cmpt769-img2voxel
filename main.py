import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import cv2
import os
from depth2normal import get_surface_normal_by_depth, get_normal_map_by_point_cloud
from seg import img_seg_slic

# FX_DEPTH = 500
# FY_DEPTH = 500
# CX_DEPTH = 0
# CY_DEPTH = 0

FX_DEPTH = 5.8262448167737955e+02
FY_DEPTH = 5.8269103270988637e+02
CX_DEPTH = 3.1304475870804731e+02
CY_DEPTH = 2.3844389626620386e+02


def show_image(img):
    print(img.shape, np.min(img), np.max(img), type(img))
    plt.imshow(img)
    plt.show()


def scale_depth(depth, scale_factor=1000.0):
    depth = depth.astype(np.float32) / scale_factor
    return depth


def from_8uc1_to_16uc1(img):
    img = img.astype(np.uint16)
    return img


def from_16uc1_to_8uc1(img):
    img = img / 256
    img = img.astype(np.uint8)
    return img


def pcl_from_depth(depth, fx=FX_DEPTH, fy=FY_DEPTH, cx=CX_DEPTH, cy=CY_DEPTH):
    width, height = depth.shape
    depth = o3d.geometry.Image(depth)
    pcd = o3d.geometry.PointCloud.create_from_depth_image(depth, o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy))
    # pcd = np.asarray(pcd.points)
    return pcd


def colored_pcl_from_depth(depth, img, fx=FX_DEPTH, fy=FY_DEPTH, cx=CX_DEPTH, cy=CY_DEPTH):
    width, height = depth.shape
    depth = o3d.geometry.Image(depth)
    img = o3d.geometry.Image(img)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(o3d.geometry.RGBDImage.create_from_color_and_depth(img, depth),
                                                         o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy))
    # o3d.visualization.draw_geometries([pcd])
    # pcd = np.asarray(pcd.points)
    return pcd


def pcl_from_depth2(depth, fx=FX_DEPTH, fy=FY_DEPTH, cx=CX_DEPTH, cy=CY_DEPTH):
    width, height = depth.shape
    pcl = []
    for x in range(width):
        for y in range(height):
            z = depth[x, y]
            x_ = (x - cx) * z / fx
            y_ = (y - cy) * z / fy
            pcl.append([x_, y_, z])
    pcl = np.array(pcl)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcl)
    return pcl


def visualize_pcl(pcd):
    if type(pcd) == np.ndarray:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcl)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(pcl)
    # pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]])
    o3d.visualization.draw_geometries([pcd])


def o3d_pcl_plane_seg(pcd):
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(pcl)
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.001, ransac_n=3, num_iterations=1000)
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a}x + {b}y + {c}z + {d} = 0")

    inlier_cloud = pcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])
    outlier_cloud = pcd.select_by_index(inliers, invert=True)

    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

    return plane_model, inliers


def rotate_point_cloud_align_z_axis(pcd, plane, original_pcd=None, visualize=False):
    [a, b, c, d] = plane
    # get the normal vector of the plane
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal)

    # we need to first Compute the angle between the normal vector and the z-axis (0, 0, 1)
    angle = np.arccos(np.dot(normal, np.array([0, 0, 1])))
    # print("Angle between the normal vector and the z-axis: ", angle)

    # finding the rotation axis
    axis = np.cross(normal, np.array([0, 0, 1]))
    axis = axis / np.linalg.norm(axis)

    # finding the rotation matrix using the Rodrigues formula
    R = np.array([[np.cos(angle) + axis[0] ** 2 * (1 - np.cos(angle)), axis[0] * axis[1] * (1 - np.cos(angle)) - axis[2] * np.sin(angle), axis[0] * axis[2] * (1 - np.cos(angle)) + axis[1] * np.sin(angle)],
                  [axis[1] * axis[0] * (1 - np.cos(angle)) + axis[2] * np.sin(angle), np.cos(angle) + axis[1] ** 2 * (1 - np.cos(angle)), axis[1] * axis[2] * (1 - np.cos(angle)) - axis[0] * np.sin(angle)],
                  [axis[2] * axis[0] * (1 - np.cos(angle)) - axis[1] * np.sin(angle), axis[2] * axis[1] * (1 - np.cos(angle)) + axis[0] * np.sin(angle), np.cos(angle) + axis[2] ** 2 * (1 - np.cos(angle))]])

    # rotate the point cloud
    points_before_rotation = np.asarray(pcd.points)
    pcd.rotate(R, center=(0, 0, 0))

    if visualize:
        # visualize the rotated point cloud
        o3d.visualization.draw_geometries([pcd, original_pcd])

        # rerotate the point cloud back to the original position
        pcd.rotate(R.T, center=(0, 0, 0))
        # visualize the rotated point cloud
        o3d.visualization.draw_geometries([pcd, original_pcd])
        points_after_rotation = np.asarray(pcd.points)

        # compute the error between the original point cloud and the rotated point cloud
        error = np.sum(np.abs(points_before_rotation - points_after_rotation))
        print("Error: ", error)

    return pcd


def o3d_pcl_multi_plane_seg(pcd, num_plane=3, thr=0.01):
    for i in range(num_plane):
        plane_model, inliers = pcd.segment_plane(distance_threshold=thr, ransac_n=3, num_iterations=1000)
        [a, b, c, d] = plane_model
        print(f"Plane equation: {a}x + {b}y + {c}z + {d} = 0")
        inlier_cloud = pcd.select_by_index(inliers)
        inlier_cloud.paint_uniform_color([1.0, 0, 0])
        outlier_cloud = pcd.select_by_index(inliers, invert=True)
        o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
        pcd = outlier_cloud

        x = 0


# def o3d_get_cuboids_from_pcd(pcd, num_plane=3, ransac_n=3, dist_threshold=0.01, max_iterations=1000):
#     # pcd = o3d.geometry.PointCloud()
#     # pcd.points = o3d.utility.Vector3dVector(pcl)
#
#     for i in range(num_plane):
#

# Fit cuboids to each plane
#     cuboids = []
#
#     # Extract the plane normal vector
#     normal_vector = np.array(plane_model[0:3])
#
#     # Orient the plane so that its normal vector is aligned with the z-axis
#     R = o3d.geometry.get_rotation_matrix_from_xyz(normal_vector)
#     pcd.rotate(R, center=normal_vector)
#
#     # Project points onto the plane and compute the 2D bounding box
#     projected_points = np.delete(pcd.points, inliers[i], axis=0) - plane_model[0:3]
#     x_min, y_min = np.min(projected_points[:, [0, 1]], axis=0)
#     x_max, y_max = np.max(projected_points[:, [0, 1]], axis=0)
#
#     # Extrude the 2D bounding box to create a cuboid with random orientation
#     x_extent = x_max - x_min
#     y_extent = y_max - y_min
#     z_extent = np.max(pcd.points[:, 2]) - np.min(pcd.points[:, 2])
#     cuboid = o3d.geometry.OrientedBoundingBox(center=plane_model[0:3],
#                                               extent=np.array([x_extent, y_extent, z_extent]))
#     cuboid.color = np.array([0.1, 0.1, 0.8])
#     R_cuboid = o3d.geometry.get_rotation_matrix_from_xyz(np.random.uniform(-np.pi, np.pi, size=3))
#     cuboid.rotate(R_cuboid, center=plane_model[0:3])
#     cuboids.append(cuboid)
#
#     # Remove the points that are inside the cuboid
#     outlier_cloud = pcd.select_by_index(inliers, invert=True)
#     pcd = outlier_cloud
#
# # Visualize the point cloud and the fitted cuboids
# o3d.visualization.draw_geometries([pcd] + cuboids)


if __name__ == '__main__':
    img_dir = './data'
    # img_id = '0024'
    # depth_dir = 'SI_R20'
    # rgb_dir = 'RGB'
    # img_id = '0022'
    # depth_dir = 'SI_R20_lowres'
    # rgb_dir = 'RGB_lowres'
    img_id = '15'
    depth_dir = 'NYU-Depth'
    rgb_dir = 'NYU-RGB'

    # img = cv2.imread(os.path.join(img_dir, rgb_dir, img_id + '_rgb.png'))
    # depth = cv2.imread(os.path.join(img_dir, depth_dir, img_id + '_final_midas_v2_o2m.png'), cv2.IMREAD_ANYDEPTH)

    # img = cv2.imread(os.path.join(img_dir, rgb_dir, img_id + '_rgb_lowres.png'))
    # depth = cv2.imread(os.path.join(img_dir, depth_dir, img_id + '_final_midas_v2_o2m_lowres.png'), cv2.IMREAD_ANYDEPTH)
    # # depth = from_16uc1_to_8uc1(depth)
    # depth = (depth / 256.).astype(np.uint16)
    # cv2.imwrite(os.path.join(img_id + '_final_midas_v2_o2m_lowres.png'), depth)

    img = cv2.imread(os.path.join(img_dir, rgb_dir, img_id + '.jpg'))
    depth = cv2.imread(os.path.join(img_dir, depth_dir, img_id + '.png'), cv2.IMREAD_ANYDEPTH)
    depth = from_8uc1_to_16uc1(depth)
    # cv2.imwrite(os.path.join('15.png'), depth)

    show_image(depth)

    # depth = scale_depth(depth, scale_factor=65535.0)
    # show_image(depth)

    # pcl = pcl_from_depth(depth)
    pcl = colored_pcl_from_depth(depth, img)
    # pcl = pcl_from_depth2(depth)
    visualize_pcl(pcl)

    # o3d_pcl_plane_seg(pcl)
    o3d_pcl_multi_plane_seg(pcl, num_plane=5, thr=0.002)

    # K = np.array([[FX_DEPTH, 0, CX_DEPTH], [0, FY_DEPTH, CY_DEPTH], [0, 0, 1]])
    K = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    normal1 = get_surface_normal_by_depth(depth, K)
    # normal1 = get_surface_normal_by_depth(depth)
    normal2, pcl = get_normal_map_by_point_cloud(depth, K)
    vis_normal = lambda normal: np.uint8((normal + 1) / 2 * 255)[..., ::-1]
    normal1 = vis_normal(normal1)
    normal2 = vis_normal(normal2)
    show_image(normal1)
    show_image(normal1[..., 0])
    show_image(normal1[..., 1])
    show_image(normal1[..., 2])
    show_image(normal2)

    seg = img_seg_slic(img, n_segments=20, compactness=10, sigma=0)
    show_image(seg)
    seg = img_seg_slic(normal1, n_segments=20, compactness=10, sigma=0)
    show_image(seg)
    seg = img_seg_slic(np.stack([depth, depth, depth], axis=-1), n_segments=20, compactness=10, sigma=0)
    show_image(seg)
    seg = img_seg_slic(np.stack([normal1[..., 0], normal1[..., 0], normal1[..., 0]], axis=-1), n_segments=20, compactness=0.01, sigma=0)
    show_image(seg)
    seg = img_seg_slic(np.stack([normal1[..., 1], normal1[..., 1], normal1[..., 1]], axis=-1), n_segments=20, compactness=0.01, sigma=0)
    show_image(seg)
    seg = img_seg_slic(np.stack([normal1[..., 2], normal1[..., 2], normal1[..., 2]], axis=-1), n_segments=20, compactness=0.01, sigma=0)
    show_image(seg)

    # visualize_pcl(pcl)
