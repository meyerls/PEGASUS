#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2022 Lukas Meyer
Licensed under the MIT License.
See LICENSE file for more information.
"""

from copy import deepcopy

import numpy as np
import open3d as o3d
import exiftool
import matplotlib.pyplot as plt

def get_labeled_exif(exif):
    labeled = {}
    for (key, val) in exif.items():
        labeled[TAGS.get(key)] = val
    return labeled


def load_image_meta_data(image_path: str) -> np.ndarray:
    """
    Load Exif meta data

    :param image_path:
    :return:
    """

    with exiftool.ExifToolHelper() as et:
        metadata = et.get_metadata(image_path)

    return metadata[0]


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2]])


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def convert_colmap_extrinsics(frame):
    Rwc = frame.Rwc()
    twc = frame.twc()

    M = np.eye(4)
    M[:3, :3] = Rwc
    M[:3, 3] = twc

    return Rwc, twc, M


def generate_colmap_sparse_pc(points3D: np.ndarray) -> o3d.pybind.geometry.PointCloud:
    sparse_pc = np.zeros((points3D.__len__(), 3))
    sparse_pc_color = np.zeros((points3D.__len__(), 3))

    for idx, pc_idx in enumerate(points3D.__iter__()):
        sparse_pc[idx] = points3D[pc_idx].xyz
        sparse_pc_color[idx] = points3D[pc_idx].rgb / 255.

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(sparse_pc)
    pc.colors = o3d.utility.Vector3dVector(sparse_pc_color)

    return pc


def plot_disparity_map(disparity_image):
    fig, ax = plt.subplots(figsize=(25, 25))
    min_depth, max_depth = np.percentile(disparity_image, [5, 90])
    disparity_image[disparity_image < min_depth] = min_depth
    disparity_image[disparity_image > max_depth] = max_depth
    disparity_image[disparity_image == 0] = max_depth + 0.1 * max_depth
    pos = ax.imshow(disparity_image, cmap="Greys")
    plt.axis('off')
    cbar = fig.colorbar(pos, ax=ax, shrink=0.52)
    cbar.ax.tick_params(size=0)
    cbar.set_ticks([])
    plt.show()


def plot_alpha_blending(rgb_image, disparity_image):
    rgb_image = deepcopy(rgb_image)
    disparity_image = deepcopy(disparity_image)
    fig, ax = plt.subplots(figsize=(15, 15))
    min_depth, max_depth = np.percentile(disparity_image, [5, 95])
    disparity_image[disparity_image < min_depth] = min_depth
    disparity_image[disparity_image > max_depth] = max_depth
    disparity_image[disparity_image == 0] = max_depth + 0.1 * max_depth
    pos = ax.imshow(disparity_image, cmap="Greys")
    plt.axis('off')

    pos = ax.imshow(rgb_image, alpha=0.3)  # interpolation='none'plt.imshow(dst)
    plt.show()
