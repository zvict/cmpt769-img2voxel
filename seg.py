import cv2
import os
import numpy as np
import skimage


def img_seg_slic(img, n_segments=100, compactness=10, sigma=1):
    # img = cv2.imread(img_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = skimage.img_as_float(img)
    segments = skimage.segmentation.slic(img, n_segments=n_segments, compactness=compactness, sigma=sigma)
    return segments