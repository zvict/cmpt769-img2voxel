import imageio.v3 as iio
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import open3d as o3d
import torch
import cv2
import os
import skimage
import sklearn.cluster as skc
from sklearn.cluster import MeanShift, estimate_bandwidth
from depth2normal import get_surface_normal_by_depth, get_normal_map_by_point_cloud
from main import *
from point2plane import *


MIN_PTS_IN_CLUSTER = 500
NORMAL_EST_RADIUS = 100
NORMAL_EST_KNN = 300
NORMAL_EST_SCALE = 10000


def filter_out_small_clusters(cur_points, cur_labels, min_pts=MIN_PTS_IN_CLUSTER):
    filtered_labels = []
    filtered_points = []
    cnt = 0
    for v in range(cur_labels.min(), cur_labels.max() + 1):
        print("cur label: ", v)
        filter_mask = (cur_labels == v)
        if filter_mask.sum() >= min_pts:
            print("filter_mask: ", filter_mask.shape, filter_mask.sum())
            filtered_labels.append(np.ones_like(cur_labels[filter_mask]) * cnt)
            filtered_points.append(cur_points[filter_mask, :])
            cnt += 1 

    cur_labels = np.concatenate(filtered_labels, axis=0)
    cur_points = np.concatenate(filtered_points, axis=0)
    return cur_points, cur_labels


def sk_k_means_img(img, n_clusters=2):
    H, W, C = img.shape
    img = skimage.img_as_float(img)
    img = img.reshape((-1, C))
    k_means = skc.KMeans(n_clusters=n_clusters)
    k_means.fit(img)
    labels = k_means.labels_
    cluster_centers = k_means.cluster_centers_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    print("number of estimated clusters : %d" % n_clusters_)
    # print("cluster centers: ", cluster_centers)
    # print("labels: ", labels)
    segmented_img = cluster_centers[labels.reshape(H, W)]
    # segmented_img = cluster_centers[labels]
    return segmented_img, labels.reshape(H, W)


def sk_k_means(feat, n_clusters=2):
    if feat.ndim == 1:
        feat = feat.reshape((-1, 1))
    k_means = skc.KMeans(n_clusters=n_clusters)
    k_means.fit(feat)
    labels = k_means.labels_
    cluster_centers = k_means.cluster_centers_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    print("number of estimated clusters : %d" % n_clusters_)
    # print("cluster centers: ", cluster_centers)
    # print("labels: ", labels)
    return labels


def sk_mean_shift(feat, bandwidth=None):
    if feat.ndim == 1:
        feat = feat.reshape((-1, 1))
    if bandwidth is None:
        bandwidth = estimate_bandwidth(feat, quantile=0.1, n_samples=100)
        print("estimated bandwidth: ", bandwidth)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(feat)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    print("number of estimated clusters : %d" % n_clusters_)
    # print("cluster centers: ", cluster_centers)
    # print("labels: ", labels)
    return labels


def sk_dbscan(feat, eps=0.5, min_samples=5):
    if feat.ndim == 1:
        feat = feat.reshape((-1, 1))
    db = skc.DBSCAN(eps=eps, min_samples=min_samples).fit(feat)
    labels = db.labels_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    print("number of estimated clusters : %d" % n_clusters_)
    # print("cluster centers: ", cluster_centers)
    # print("labels: ", labels)
    return labels


def sk_spectral_clustering(feat, n_clusters=2):
    if feat.ndim == 1:
        feat = feat.reshape((-1, 1))
    sc = skc.SpectralClustering(n_clusters=n_clusters)
    sc.fit(feat)
    labels = sc.labels_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    print("number of estimated clusters : %d" % n_clusters_)
    # print("cluster centers: ", cluster_centers)
    # print("labels: ", labels)
    return labels


depth = iio.imread("./data/NYU-Depth/15.png")
normal = iio.imread("data/seg-nyu15-normal1-sigma6.png")
# show_image(normal)
# show_image(depth)
# exit(0)
H, W, C = normal.shape
normal = (normal / 255.0 * 2 - 1).astype(np.float32)
# normal = normal / np.linalg.norm(normal, axis=-1, keepdims=True)
normal_dirs = np.unique(normal.reshape((-1, C)), axis=0)
print(normal_dirs)
mask = (normal == normal_dirs[0]).all(axis=-1)
# show_image(mask)
# exit(0)


pcl = colored_pcl_from_depth(depth, normal)
aabb = pcl.get_axis_aligned_bounding_box()
# pcl.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
points = np.asarray(pcl.points) * NORMAL_EST_SCALE
colors = np.asarray(pcl.colors)
print(points.shape, points.min(), points.max())

axis_dirs = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]).astype(np.float32)


cmap = matplotlib.cm.get_cmap('Spectral')
all_points = []
all_colors = []
all_labels = []
all_normals = []
label_cnt = 0
for i, dir in enumerate(normal_dirs):
    print("---cur normal dir: ", dir)
    cur_mask = (normal == dir).all(axis=-1).reshape(-1)
    cur_points = points[cur_mask, :]
    print(cur_points.shape)

    # Step 1: filtering out small point clouds by dbscan
    # cur_labels = sk_k_means(cur_points, n_clusters=50)
    # cur_labels = sk_mean_shift(cur_points)
    cur_labels = sk_dbscan(cur_points, eps=0.1, min_samples=10)
    print("cur_labels: ", cur_labels.shape)

    cur_points, cur_labels = filter_out_small_clusters(cur_points, cur_labels, MIN_PTS_IN_CLUSTER)

    # Step 2: clustering by the depth of the point cloud in the estimated normal direction
    cur_pcl = o3d.geometry.PointCloud()
    cur_pcl.points = o3d.utility.Vector3dVector(cur_points)

    cur_pcl.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=NORMAL_EST_RADIUS, max_nn=NORMAL_EST_KNN))
    # cur_pcl.estimate_normals()
    cur_normals = np.asarray(cur_pcl.normals)
    print("cur_normals: ", cur_normals.shape)
    # cur_dir = np.sum(cur_normals, axis=0)
    cur_dir = np.mean(cur_normals, axis=0)
    cur_dir = cur_dir / np.linalg.norm(cur_dir)
    if np.dot(cur_dir, dir) > 0:
        cur_dir = -cur_dir
    print("cur_dir: ", cur_dir)
    cur_depth = np.dot(cur_points, cur_dir)

    cur_query = cur_depth
    # cur_query = np.concatenate([cur_points, cur_depth[:, None], cur_depth[:, None], cur_depth[:, None]], axis=-1)
    # cur_labels = sk_k_means(cur_query, n_clusters=10)
    cur_labels = sk_mean_shift(cur_query)
    # cur_labels = sk_dbscan(cur_query, eps=0.1, min_samples=10)
    print("cur_labels: ", cur_labels.shape)

    cur_points, cur_labels = filter_out_small_clusters(cur_points, cur_labels, MIN_PTS_IN_CLUSTER)

    # Step 3: for each cluster, do clustering with dbscan
    cur_label_cnt = 0
    new_points = []
    new_labels = []
    for v in range(cur_labels.min(), cur_labels.max() + 1):
        cur_mask = (cur_labels == v)
        cur_cluster = cur_points[cur_mask, :]
        cur_cluster_labels = sk_dbscan(cur_cluster, eps=0.1, min_samples=10)

        cur_cluster, cur_cluster_labels = filter_out_small_clusters(cur_cluster, cur_cluster_labels, MIN_PTS_IN_CLUSTER)

        cur_cluster_labels = cur_cluster_labels + cur_label_cnt
        new_points.append(cur_cluster)
        new_labels.append(cur_cluster_labels)
        cur_label_cnt += len(np.unique(cur_cluster_labels))
        print("cur_label_cnt: ", cur_label_cnt)

    cur_points = np.concatenate(new_points, axis=0)
    cur_labels = np.concatenate(new_labels, axis=0)

    # cur_colors = cmap((cur_depth - cur_depth.min()) / (cur_depth.max() - cur_depth.min()))[:, :3]
    cur_colors = cmap((cur_labels - cur_labels.min()) / (cur_labels.max() - cur_labels.min()))[:, :3]
    
    # cur_pcl = o3d.geometry.PointCloud()
    # cur_pcl.points = o3d.utility.Vector3dVector(cur_points)
    # # cur_pcl.colors = o3d.utility.Vector3dVector(colors[mask, :])
    # # cur_pcl.colors = o3d.utility.Vector3dVector(cmap(cur_labels / cur_labels.max())[:, :3])
    # cur_pcl.colors = o3d.utility.Vector3dVector(cur_colors)
    # visualize_pcl(cur_pcl)
    # del cur_pcl

    all_points.append(cur_points)
    all_colors.append(cur_colors)
    all_labels.append(cur_labels + label_cnt)
    all_normals.append(np.ones_like(cur_points) * cur_dir)
    # all_normals.append(np.ones_like(cur_labels) * dir)
    label_cnt += cur_labels.max() + 1

# exit(0)

all_points = np.concatenate(all_points, axis=0)
all_colors = np.concatenate(all_colors, axis=0)
all_labels = np.concatenate(all_labels, axis=0)
all_normals = np.concatenate(all_normals, axis=0)
print(np.unique(all_labels))
print("all_points: ", all_points.shape, "all_colors: ", all_colors.shape, "all_labels: ", all_labels.shape, "all_normals: ", all_normals.shape)

new_pcl = o3d.geometry.PointCloud()
new_pcl.points = o3d.utility.Vector3dVector(all_points)
# pcl.colors = o3d.utility.Vector3dVector(colors[mask, :])
# pcl.colors = o3d.utility.Vector3dVector(cmap(labels / labels.max())[:, :3])
all_colors = cmap((all_labels - all_labels.min()) / (all_labels.max() - all_labels.min()))[:, :3]
new_pcl.colors = o3d.utility.Vector3dVector(all_colors)
visualize_pcl(new_pcl)


# Step 4: estimate the plane for each cluster
all_clusters = {
    "points": [],
    "normals": [],
    "labels": [],
    "colors": [],
    "plane_points": [],
    "plane_normals": [],
    "projected_points": [],
}
all_planes = []
vis_normals = []
for v in range(all_labels.min(), all_labels.max() + 1):
    cur_mask = (all_labels == v)
    print("cur_mask: ", v, cur_mask.sum())
    cur_points = all_points[cur_mask, :].astype(np.float32)
    cur_normals = all_normals[cur_mask, :]

    # cur_pcl = o3d.geometry.PointCloud()
    # cur_pcl.points = o3d.utility.Vector3dVector(cur_points)
    # cur_pcl.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=NORMAL_EST_RADIUS, max_nn=NORMAL_EST_KNN))
    # cur_normals = np.asarray(cur_pcl.normals)
    # cur_normals = cur_normals / np.linalg.norm(cur_normals, axis=-1, keepdims=True)

    # cur_plane = estimate_plane(cur_points, cur_normals)
    # cur_plane, _ = point2plane_lstsq(cur_points)
    # cur_proj_points = project_points2plane(cur_points, cur_plane)
    # cur_plane = sks_point2plane(cur_points)
    # cur_proj_points = project_points_on_plane(cur_plane.point, cur_plane.normal, cur_points)
    cur_plane_point = cur_points.mean(axis=0)
    cur_plane_normal = cur_normals.mean(axis=0)
    cur_proj_points = project_points_on_plane(cur_plane_point, cur_plane_normal, cur_points)
    all_planes.append(cur_proj_points)

    all_clusters["points"].append(cur_points)
    all_clusters["normals"].append(cur_normals)
    all_clusters["labels"].append(v)
    all_clusters["colors"].append(all_colors[cur_mask, :])
    all_clusters["plane_points"].append(cur_plane_point)
    all_clusters["plane_normals"].append(cur_plane_normal)
    all_clusters["projected_points"].append(cur_proj_points)

    cur_vis_normal = o3d.geometry.LineSet()
    cur_vis_normal.points = o3d.utility.Vector3dVector(np.stack([cur_plane_point, cur_plane_point + cur_plane_normal * 0.5], axis=0))
    cur_vis_normal.lines = o3d.utility.Vector2iVector(np.array([[0, 1]]))
    cur_vis_normal.colors = o3d.utility.Vector3dVector(np.array([[1, 0, 0]]))
    vis_normals.append(cur_vis_normal)

torch.save(all_clusters, "nyu15-all_clusters.pth")

all_planes = np.concatenate(all_planes, axis=0)

new_pcl = o3d.geometry.PointCloud()
new_pcl.points = o3d.utility.Vector3dVector(all_planes)
# pcl.colors = o3d.utility.Vector3dVector(colors[mask, :])
# pcl.colors = o3d.utility.Vector3dVector(cmap(labels / labels.max())[:, :3])
new_pcl.colors = o3d.utility.Vector3dVector(all_colors)
# visualize_pcl(new_pcl)
o3d.visualization.draw_geometries([new_pcl] + vis_normals)