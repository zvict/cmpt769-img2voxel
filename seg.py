import cv2
import os
import numpy as np
import skimage
from sklearn.cluster import MeanShift, estimate_bandwidth


def img_seg_slic(img, n_segments=100, compactness=10, sigma=1):
    # img = cv2.imread(img_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = skimage.img_as_float(img)
    segments = skimage.segmentation.slic(img, n_segments=n_segments, compactness=compactness, sigma=sigma)
    return segments


def scikit_mean_shift(img, bandwidth=None):
    # img = cv2.imread(img_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W, C = img.shape
    img = skimage.img_as_float(img)
    img = img.reshape((-1, 3))
    if bandwidth is None:
        bandwidth = estimate_bandwidth(img, quantile=0.1, n_samples=100)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(img)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    print("number of estimated clusters : %d" % n_clusters_)
    print("cluster centers: ", cluster_centers)
    print("labels: ", labels)
    segmented_img = cluster_centers[labels.reshape(H, W)]
    # segmented_img = cluster_centers[labels]
    return segmented_img
