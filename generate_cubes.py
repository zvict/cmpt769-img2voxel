import imageio.v3 as iio
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import open3d as o3d
import torch
import cv2
import os
import skimage
from scipy.spatial import KDTree
from main import *
from point2plane import *

cmap = matplotlib.cm.get_cmap('Spectral')

MIN_PTS_IN_CLUSTER = 500


def text_3d(text, pos, direction=None, degree=0.0, font='C:\Windows\Fonts\Arial.ttf', font_size=20):
    """
    Generate a 3D text point cloud used for visualization.
    :param text: content of the text
    :param pos: 3D xyz position of the text upper left corner
    :param direction: 3D normalized direction of where the text faces
    :param degree: in plane rotation of text
    :param font: Name of the font - change it according to your system
    :param font_size: size of the font
    :return: o3d.geoemtry.PointCloud object
    """
    if direction is None:
        direction = (0., 0., 1.)

    from PIL import Image, ImageFont, ImageDraw, ImageOps
    from pyquaternion import Quaternion

    font_obj = ImageFont.truetype(font, font_size)
    font_dim = font_obj.getsize(text)

    img = Image.new('RGB', font_dim, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), text, font=font_obj, fill=(0, 0, 0))
    img = ImageOps.flip(img)
    img = np.asarray(img)
    img_mask = img[:, :, 0] < 128
    indices = np.indices([*img.shape[0:2], 1])[:, img_mask, 0].reshape(3, -1).T

    pcd = o3d.geometry.PointCloud()
    pcd.colors = o3d.utility.Vector3dVector(img[img_mask, :].astype(float) / 255.0)
    pcd.points = o3d.utility.Vector3dVector(indices / 100.0)

    raxis = np.cross([0.0, 0.0, 1.0], direction)
    if np.linalg.norm(raxis) < 1e-6:
        raxis = (0.0, 0.0, 1.0)
    trans = (Quaternion(axis=raxis, radians=np.arccos(direction[2])) *
             Quaternion(axis=direction, degrees=degree)).transformation_matrix
    trans[0:3, 3] = np.asarray(pos)
    pcd.transform(trans)
    return pcd


def check_pc_intersects(pc1, pc2, thr=0.01):
    target_pc = pc1 if pc1.shape[0] > pc2.shape[0] else pc2
    query_pc = pc2 if pc1.shape[0] > pc2.shape[0] else pc1
    tree = KDTree(target_pc)
    dist, _ = tree.query(query_pc)
    return np.any(dist <= thr)


def visualize_pc(clusters):
    points = np.concatenate(clusters['projected_points'], axis=0)
    colors = np.concatenate(clusters['colors'], axis=0)
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(points)
    pcl.colors = o3d.utility.Vector3dVector(colors)

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()
    visualizer.add_geometry(pcl)

    for i in range(len(clusters['points'])):
        pos = clusters['projected_points'][i].mean(axis=0)
        text_pcd = text_3d(str(i), pos, direction=(0, 0, 1), degree=-90, font_size=20)
        visualizer.add_geometry(text_pcd)

    visualizer.run()


clusters = torch.load("data/Clusters/nyu15-all_clusters.pth")
num_pcs = len(clusters['points'])
print(clusters.keys(), num_pcs)

# unique_normals = np.unique(np.concatenate(clusters['normals']), axis=0)
unique_normals = [np.unique(clusters['normals'][i], axis=0) for i in range(num_pcs)]
print("unique_normals:\n", np.unique(np.concatenate(unique_normals), axis=0))

filtered_clusters = {
    "points": [],
    "normals": [],
    "labels": [],
    "colors": [],
    "plane_points": [],
    "plane_normals": [],
    "projected_points": [],
}
for i in range(num_pcs):
    if clusters['points'][i].shape[0] >= MIN_PTS_IN_CLUSTER:
        filtered_clusters['points'].append(clusters['points'][i])
        filtered_clusters['normals'].append(clusters['normals'][i])
        filtered_clusters['labels'].append(clusters['labels'][i])
        filtered_clusters['colors'].append(clusters['colors'][i])
        filtered_clusters['plane_points'].append(clusters['plane_points'][i])
        filtered_clusters['plane_normals'].append(clusters['plane_normals'][i])
        filtered_clusters['projected_points'].append(clusters['projected_points'][i])
num_pcs = len(filtered_clusters['points'])
clusters = filtered_clusters

intersects = {}
for i in range(num_pcs):
    intersects[i] = [i]
    for j in range(num_pcs):
        if i != j and j not in intersects[i] and (unique_normals[i] != unique_normals[j]).all():
            print("checking", i, j)
            if check_pc_intersects(clusters['projected_points'][i], clusters['projected_points'][j], thr=5e-3):
                intersects[i].append(j)

print("intersects:\n", intersects)

visualize_pc(clusters)