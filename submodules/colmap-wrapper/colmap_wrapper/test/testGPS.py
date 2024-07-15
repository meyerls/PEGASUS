#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2022 Lukas Meyer
Licensed under the MIT License.
See LICENSE file for more information.
"""
import copy

import numpy as np
import pylab as plt
import open3d as o3d

# Own Modules
from colmap_wrapper.gps.visualization import GPSVisualization
from colmap_wrapper.colmap.colmap import COLMAP
from colmap_wrapper.visualization import ColmapVisualization
from colmap_wrapper import USER_NAME


def points2pcd(points: np.ndarray):
    '''
    Convert a numpy array to an open3d point cloud. Just for convenience to avoid converting it every single time.
    Assigns blue color uniformly to the point cloud.

    :param points: Nx3 array with xyz location of points
    :return: a blue open3d.geometry.PointCloud()
    '''

    colors = [[0, 0, 1] for i in range(points.shape[0])]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def kabsch_umeyama(pointset_A, pointset_B):
    """
    Kabsch–Umeyama algorithm is a method for aligning and comparing the similarity between two sets of points.
    It finds the optimal translation, rotation and scaling by minimizing the root-mean-square deviation (RMSD)
    of the point pairs.

    Source and Explenation: https://zpl.fi/aligning-point-patterns-with-kabsch-umeyama-algorithm/

    @param pointset_A: array of a set of points in n-dim
    @param pointset_B: array of a set of points in n-dim
    @return: Rotation Matrix (3x3), scaling (scalar) translation vector (3x1)
    """
    assert pointset_A.shape == pointset_B.shape
    n, m = pointset_A.shape

    # Find centroids of both point sets
    EA = np.mean(pointset_A, axis=0)
    EB = np.mean(pointset_B, axis=0)

    VarA = np.mean(np.linalg.norm(pointset_A - EA, axis=1) ** 2)

    # Covariance matrix
    H = ((pointset_A - EA).T @ (pointset_B - EB)) / n

    # SVD H = UDV^T
    U, D, VT = np.linalg.svd(H)

    # Detect and prevent reflection
    d = np.sign(np.linalg.det(U) * np.linalg.det(VT))
    S = np.diag([1] * (m - 1) + [d])

    # rotation, scaling and translation
    R = U @ S @ VT
    c = VarA / np.trace(np.diag(D) @ S)
    t = EA - c * R @ EB

    return R, c, t


def get_cartesian(lat=None, lon=None, elevation=None):
    """
    https://itecnote.com/tecnote/python-how-to-convert-longitudelatitude-elevation-to-cartesian-coordinates/
    """
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    R = 6378137.0 + elevation  # radius of the earth
    x = R * np.cos(lat) * np.cos(lon)
    y = R * np.cos(lat) * np.sin(lon)
    z = R * np.sin(lat)
    return x, y, z


def visualize_gps_osm(project, map_path, osm_boarders, save_as: str, show: bool = False):
    gps_data = {}
    extrinsics = {}
    for model_idx, model in enumerate(project.projects):
        gps_data.update({model_idx: []})
        extrinsics.update({model_idx: []})
        for image_idx in model.images.keys():
            gps_data[model_idx].append([float(model.images[image_idx].exifdata["XMP:GPSLatitude"]),
                                        float(model.images[image_idx].exifdata["XMP:GPSLongitude"]),
                                        float(model.images[image_idx].exifdata["XMP:AbsoluteAltitude"])])
            extrinsics[model_idx].append(model.images[image_idx].extrinsics)

    # Show gps --> x and y
    gps_certesian_0 = np.asarray([get_cartesian(lat, long, elev) for lat, long, elev in gps_data[0]])
    plt.scatter(gps_certesian_0[:, 0], gps_certesian_0[:, 1])

    gps_certesian_1 = np.asarray([get_cartesian(lat, long, elev) for lat, long, elev in gps_data[1]])
    plt.scatter(gps_certesian_1[:, 0], gps_certesian_1[:, 1])
    plt.show()

    # Show extrinsics (translational part)
    trans_certesian_0 = np.asarray([[array[:3, -1][0], array[:3, -1][1], array[:3, -1][2]] for array in extrinsics[0]])
    plt.scatter(trans_certesian_0[:, 0], trans_certesian_0[:, 1])

    trans_certesian_1 = np.asarray([[array[:3, -1][0], array[:3, -1][1], array[:3, -1][2]] for array in extrinsics[1]])
    plt.scatter(trans_certesian_1[:, 0], trans_certesian_1[:, 1])
    plt.show()

    gps_data_osm = []
    for key in gps_data.keys():
        gps_data_osm.extend(gps_data[key])

    vis = GPSVisualization(gps_data_osm, map_path=map_path,
                           points=osm_boarders)  # Draw converted records to the map image.

    vis.create_image(color=(0, 0, 255), width=0.2)
    vis.plot_map(show=True, save_as=save_as)

    return [gps_certesian_0, gps_certesian_1], [trans_certesian_0, trans_certesian_1]


if __name__ == '__main__':
    if False:
        path = "/home/{}/Documents/reco/23_04_14/01".format(USER_NAME)
        img_orig = None
    else:
        path = '/media/{}/Samsung_T5/For5G/reco/23_03_17/03'.format(USER_NAME)
        img_orig = '/media/{}/Samsung_T5/For5G/data/23_03_17/03'.format(USER_NAME)
    project = COLMAP(project_path=path,
                     exif_read=True,
                     img_orig=img_orig,
                     dense_pc='cropped.ply')

    colmap_project = project.project

    camera = colmap_project.cameras
    images = colmap_project.images
    sparse = colmap_project.get_sparse()
    dense = colmap_project.get_dense()

    for COLMAP_MODEL in project.projects:
        project_vs = ColmapVisualization(colmap=COLMAP_MODEL, bg_color=np.asarray([0, 0, 0]))
        project_vs.visualization(frustum_scale=0.4, image_type='image', point_size=0.001)

    gps, extrinsic = visualize_gps_osm(project=project,
                                       map_path='../gps/map.png',
                                       osm_boarders=(49.66721, 11.32313, 49.66412, 11.32784),
                                       save_as='./result_map.png',
                                       show=True)

    gps_0 = gps[0] - np.mean(gps[0], axis=0)
    gps_1 = gps[1] - np.mean(gps[0], axis=0)

    R_0, c_0, t_0 = kabsch_umeyama(gps_0, extrinsic[0])
    R_1, c_1, t_1 = kabsch_umeyama(gps_1, extrinsic[1])

    # Plot GPS
    gps_pcd_0 = points2pcd((np.asarray(gps_0)))
    gps_pcd_0.paint_uniform_color((1, 0, 0))
    gps_pcd_1 = points2pcd((np.asarray(gps_1)))
    gps_pcd_1.paint_uniform_color((0, 0, 1))
    o3d.visualization.draw_geometries([gps_pcd_0, gps_pcd_1])

    # Plot Camera
    extrinsic_pcd_0 = points2pcd((np.asarray(extrinsic[0])))
    extrinsic_pcd_0.paint_uniform_color((0, 1, 0))
    extrinsic_pcd_1 = points2pcd((np.asarray(extrinsic[1])))
    extrinsic_pcd_1.paint_uniform_color((0, 1, 1))
    o3d.visualization.draw_geometries([extrinsic_pcd_0, extrinsic_pcd_1])

    extrinsic_pcd_oriented_0 = copy.deepcopy(extrinsic_pcd_0)
    extrinsic_pcd_oriented_1 = copy.deepcopy(extrinsic_pcd_1)

    I_0 = np.eye(4)
    I_0[:3, :3] = c_0 * R_0
    I_0[:3, -1] = t_0

    extrinsic_pcd_oriented_0 = extrinsic_pcd_oriented_0.transform(I_0)

    # extrinsic_pcd_oriented_0 = extrinsic_pcd_oriented_0.rotate(R_0)
    # extrinsic_pcd_oriented_0 = extrinsic_pcd_oriented_0.scale(c_0, center=(0, 0, 0))
    # extrinsic_pcd_oriented_0 = extrinsic_pcd_oriented_0.translate(t_0)

    I_1 = np.eye(4)
    I_1[:3, :3] = c_1 * R_1
    I_1[:3, -1] = t_1

    extrinsic_pcd_oriented_1 = extrinsic_pcd_oriented_1.transform(I_1)

    # extrinsic_pcd_oriented_1 = extrinsic_pcd_oriented_1.rotate(R_1)
    # extrinsic_pcd_oriented_1 = extrinsic_pcd_oriented_1.scale(c_1, center=(0, 0, 0))
    # extrinsic_pcd_oriented_1 = extrinsic_pcd_oriented_1.translate(t_1)

    # Aligned gps and poses
    o3d.visualization.draw_geometries([extrinsic_pcd_oriented_0, extrinsic_pcd_oriented_1, gps_pcd_0, gps_pcd_1])

    # Align dense pcd
    side_0 = copy.deepcopy(project.projects[0].dense)
    side_1 = copy.deepcopy(project.projects[1].dense)

    # o3d.visualization.draw_geometries([side_0, side_1, extrinsic_pcd_0, extrinsic_pcd_1])

    side_0 = side_0.transform(I_0)
    # side_0 = side_0.rotate(R_0)
    # side_0 = side_0.scale(c_0, center=(0, 0, 0))
    # side_0 = side_0.translate(t_0)

    side_1 = side_1.transform(I_1)
    # side_1 = side_1.rotate(R_1)
    # side_1 = side_1.scale(c_1, center=(0, 0, 0))
    # side_1 = side_1.translate(t_1)
    o3d.visualization.draw_geometries([side_0, side_1])

    # project.projects[1].dense = project.projects[1].dense.scale(c_1)

    o3d.visualization.draw_geometries(
        [side_0, side_1, extrinsic_pcd_oriented_0, extrinsic_pcd_oriented_1, gps_pcd_0, gps_pcd_1])

    o3d.io.write_point_cloud('side_0.ply', side_0)
    o3d.io.write_point_cloud('side_1.ply', side_1)

    for COLMAP_MODEL in project.projects:
        project_vs = ColmapVisualization(colmap=COLMAP_MODEL, bg_color=np.asarray([0, 0, 0]))
        project_vs.visualization(frustum_scale=0.4, image_type='image', point_size=0.001)

    print('Finished')
