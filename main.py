import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import cv2
import os
from depth2normal import get_surface_normal_by_depth, get_normal_map_by_point_cloud
from seg import *
from utils import *


# FX_DEPTH = 2000
# FY_DEPTH = 2000
# CX_DEPTH = 300
# CY_DEPTH = 200

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
    pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]])
    # o3d.visualization.draw_geometries([pcd])
    # opt = vis.get_render_option()
    # opt.show_coordinate_frame = True
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    opt.show_coordinate_frame = True
    vis.run()


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


def o3d_pcl_multi_plane_seg(pcd, num_plane=3, thr=0.01):
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(pcl)
    for i in range(num_plane):
        plane_model, inliers = pcd.segment_plane(distance_threshold=thr, ransac_n=3, num_iterations=1000)
        [a, b, c, d] = plane_model
        print(f"Plane equation: {a}x + {b}y + {c}z + {d} = 0")

        inlier_cloud = pcd.select_by_index(inliers)
        inlier_cloud.paint_uniform_color([1.0, 0, 0])
        outlier_cloud = pcd.select_by_index(inliers, invert=True)

        o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

        pcd = outlier_cloud


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
    # depth = from_16uc1_to_8uc1(depth)
    # depth = (depth / 256.).astype(np.uint16)
    # cv2.imwrite(os.path.join(img_id + '_final_midas_v2_o2m_lowres.png'), depth)

    img = cv2.imread(os.path.join(img_dir, rgb_dir, img_id + '.jpg'))
    depth = cv2.imread(os.path.join(img_dir, depth_dir, img_id + '.png'), cv2.IMREAD_ANYDEPTH)
    depth = from_8uc1_to_16uc1(depth)
    normal = iio.imread("seg-nyu15-normal1-sigma6.png")
    # cv2.imwrite(os.path.join('15.png'), depth)

    # show_image(depth)

    # depth = scale_depth(depth, scale_factor=65535.0)
    depth = scale_depth(depth, scale_factor=255.0)
    show_image(depth)

    # pcl = pcl_from_depth(depth)
    # pcl = colored_pcl_from_depth(depth, img)
    # # pcl = pcl_from_depth2(depth)
    # visualize_pcl(pcl)
    # o3d_pcl_plane_seg(pcl)
    # o3d_pcl_multi_plane_seg(pcl, num_plane=5, thr=0.002)

    K = np.array([[FX_DEPTH, 0, CX_DEPTH], [0, FY_DEPTH, CY_DEPTH], [0, 0, 1]])
    # K = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    normal1 = get_surface_normal_by_depth(depth, K)
    # normal1 = get_surface_normal_by_depth(depth)
    normal2, pcl = get_normal_map_by_point_cloud(depth, K)
    vis_normal = lambda normal: np.uint8((normal + 1) / 2 * 255)[..., ::-1]
    normal1 = vis_normal(normal1)
    normal2 = vis_normal(normal2)

    show_image(normal1)
    # show_image(normal1[..., 0])
    # show_image(normal1[..., 1])
    # show_image(normal1[..., 2])
    show_image(normal2)

    # seg = img_seg_slic(img, n_segments=20, compactness=10, sigma=0)
    # seg = scikit_mean_shift(img)
    # show_image(seg)
    # seg = img_seg_slic(normal1, n_segments=20, compactness=10, sigma=0)
    # seg = scikit_mean_shift(normal1)
    # show_image(seg)
    # seg = img_seg_slic(np.stack([depth, depth, depth], axis=-1), n_segments=20, compactness=10, sigma=0)
    # seg = img_seg_slic(np.stack([normal1[..., 0], normal1[..., 0], normal1[..., 0]], axis=-1), n_segments=20, compactness=0.01, sigma=0)
    # show_image(seg)
    # seg = img_seg_slic(np.stack([normal1[..., 1], normal1[..., 1], normal1[..., 1]], axis=-1), n_segments=20, compactness=0.01, sigma=0)
    # show_image(seg)
    # seg = img_seg_slic(np.stack([normal1[..., 2], normal1[..., 2], normal1[..., 2]], axis=-1), n_segments=20, compactness=0.01, sigma=0)
    # show_image(seg)

    # query = img
    # query = np.stack([depth, depth, depth], axis=-1)
    # query = normal1
    # query = normal1 / 255.
    # show_image(query)

    # query = skimage.filters.gaussian(query, sigma=6)
    # show_image(query)

    # # # seg = scikit_mean_shift(img)
    # seg = scikit_mean_shift(query)
    # # # seg = scikit_mean_shift(np.stack([depth, depth, depth], axis=-1))
    # seg = (seg * 255).astype(np.uint8)
    # show_image(seg)
    # print(seg.shape, seg.dtype)
    # iio.imwrite('seg-nyu15-normal1-sigma6.png', seg)

    # segs = []
    # for sigma in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    #     cur_query = skimage.filters.gaussian(query, sigma=sigma)
    #     seg = scikit_mean_shift(cur_query)
    #     segs.append(seg)
    # plot_images(segs)

    # segs = []
    # for ksize in [3, 5, 7, 9, 11, 13]:
    #     cur_query = skimage.filters.laplace(query, ksize=ksize)
    #     # show_image(cur_query)
    #     seg = scikit_mean_shift(cur_query)
    #     segs.append(seg)
    # plot_images(segs)


    # pcl = colored_pcl_from_depth(depth, normal)
    # # pcl = pcl_from_depth2(depth)
    # visualize_pcl(pcl)
