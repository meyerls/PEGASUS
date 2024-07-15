#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import sys

sys.path.append("/gaussian-splatting")

import torch
from scene import Scene
from gaussian_renderer import render, network_gui
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import math as m
import numpy as np
import json
from scipy.spatial.transform import Rotation
import copy
import src.dataset.dataset_envs as env_assets
import src.dataset_objects as object_assets
from src.utility.pose_interpolation import interpolate_pose
from scene.cameras import Camera
# from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary
from utils.graphics_utils import focal2fov, getWorld2View2, getWorld2View
from pathlib import Path
import pylab as plt
import cv2
import os
from utils.sh_utils import RGB2SH, SH2RGB
import colorsys

sys.path.append("C:/Users/meyerls/Documents/AIST/code/gaussian-splatting/colmap-wrapper")

from colmap_wrapper.dataloader import (write_images_binary, write_points3D_binary, write_cameras_binary,
                                       write_cameras_text, write_images_text, write_points3D_text,
                                       read_images_binary, read_points3d_binary, read_cameras_binary)

def generate_colors(n):
    colors = []

    for i in range(n):
        hue = i / n  # Vary the hue based on the index
        saturation = 0.7  # adjust the saturation and lightness as needed
        lightness = 0.6

        rgb_color = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors.append(tuple(c for c in rgb_color))

    return torch.asarray(colors, dtype=torch.float32).to('cuda')

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


def gaussian_splatting_manipulation(physics_file: str,
                                    dataset_path: str,
                                    load_iteration: int = 30_000,
                                    IP: str = "127.0.0.1",
                                    PORT: int = 6009,
                                    QUIET: bool = False):
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    with open(physics_file) as data_file:
        data_loaded = json.load(data_file)

    environment_name = list(data_loaded['asset_infos']['environment'].keys())[0]
    env_class_name = data_loaded['asset_infos']['environment'][environment_name]['class_name']
    env = getattr(env_assets, env_class_name)(dataset_path=dataset_path)

    sys.argv.append('-m')
    sys.argv.append(env.gs_model_path)

    args = get_combined_args(parser)

    print("Rendering Environment" + env.reconstruction_path)

    camera_json_path = Path(env.gs_model_path) / 'cameras.json'
    with open(camera_json_path, 'r') as f:
        camera_data = json.load(f)

    cam_extr = read_images_binary(Path(env.reconstruction_path) / 'sparse/0/images.bin')
    cam_intr = read_cameras_binary(Path(env.reconstruction_path) / 'sparse/0/cameras.bin')

    network_gui.init(IP, PORT)

    cams = []

    for pose_idx in range(0, 5):

        idx = sorted(cam_extr.keys())[pose_idx]
        idx_next = sorted(cam_extr.keys())[pose_idx + 1]

        pose1 = np.eye(4)
        R = qvec2rotmat(cam_extr[idx].qvec)
        t = np.array(cam_extr[idx].tvec)
        pose1[:3, :3] = R
        pose1[:3, 3] = t

        pose2 = np.eye(4)
        R = qvec2rotmat(cam_extr[idx_next].qvec)
        t = np.array(cam_extr[idx_next].tvec)
        pose2[:3, :3] = R
        pose2[:3, 3] = t

        for frame_idx in np.linspace(0, 1, 70)[:-1]:
            T = interpolate_pose(t=frame_idx, t1=0, pose1=pose1, t2=1, pose2=pose2)

            height = 1067
            width = 1600

            FovY = focal2fov(np.asarray(camera_data[0]['fy']), cam_intr[1].height)
            FovX = focal2fov(np.asarray(camera_data[0]['fx']), cam_intr[1].width)

            R = np.transpose(T[:3, :3])
            t = np.array(T[:3, 3])

            # FovY = focal2fov(np.asarray(camera_data[0]['fy']), height)
            # FovX = focal2fov(np.asarray(camera_data[0]['fx']), width)
            viewpoint_cam = Camera(colmap_id=1,
                                   R=R,
                                   # T=np.asarray(camera_data[0]['position']),
                                   T=t,
                                   FoVx=FovX,
                                   FoVy=FovY,
                                   image=torch.empty((3, height, width)),
                                   gt_alpha_mask=None,
                                   image_name='interpolated',
                                   uid=0,
                                   data_device='cuda')
            cams.append(viewpoint_cam)

    # Initialize system state (RNG)
    safe_state(QUIET)
    pipe = pipeline.extract(args)
    # render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)

    dataset = model.extract(args)

    if True:
        video_name = 'segmentation.mp4'
        video = cv2.VideoWriter(video_name, 0, 20, (width, height))

    with torch.no_grad():
        gaussian_environment = GaussianModel(dataset.sh_degree)
        gaussian_environment.load_ply(env.gaussian_point_cloud_path(iteration=load_iteration))
        gaussian_environment.mask_points(torch.zeros(gaussian_environment._xyz.shape[0], dtype=bool).to('cuda'))

        # scene = Scene(dataset, gaussian_environment, load_iteration=load_iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        gaussians_object_list = {}

        n = 0
        for object_name in data_loaded['asset_infos']['object'].keys():
            for id in data_loaded['asset_infos']['object'][object_name]['id']:
                n += 1
        generated_colors = generate_colors(n)


        for object_name, semantic_color in zip(data_loaded['asset_infos']['object'].keys(), generated_colors ):
            for id in data_loaded['asset_infos']['object'][object_name]['id']:
                obj_class_name = data_loaded['asset_infos']['object'][object_name]['class_name']
                obj = getattr(object_assets, obj_class_name)(dataset_path=dataset_path)

                obj.mode = 'fused'
                gs_object = GaussianModel(dataset.sh_degree)
                gs_object.load_ply(obj.gaussian_point_cloud_path(iteration=load_iteration))

                base_color = RGB2SH(semantic_color)
                gs_object._features_dc[:] = base_color
                gs_object._features_rest[:, :] = torch.asarray([0, 0, 0])

                t_init = torch.from_numpy(-np.asarray(data_loaded['trajectory'][str(id)][str(0)]['t'])).type(
                    torch.float32).to(
                    device="cuda")
                gs_object.apply_translation_on_xyz(t=t_init)

                gaussians_object_list.update({id: gs_object})

        if network_gui.conn is None:
            network_gui.try_connect()

        i = 1
        gs_base = copy.deepcopy(gaussian_environment)

        while network_gui.conn is not None:

            if i < data_loaded['trajectory'][str(0)].__len__():
                gaussian_scene = copy.deepcopy(gs_base)


                for gs_object_id in gaussians_object_list.keys():
                    t_delta = torch.from_numpy(
                        np.asarray(data_loaded['trajectory'][str(gs_object_id)][str(i - 1)]['t']) - np.asarray(
                            data_loaded['trajectory'][str(gs_object_id)][str(i)]['t'])).type(
                        torch.float32).to(device="cuda")
                    q_t_1 = Rotation.from_quat(np.asarray(data_loaded['trajectory'][str(gs_object_id)][str(i)]['q']))
                    q_t_0 = Rotation.from_quat(
                        np.asarray(data_loaded['trajectory'][str(gs_object_id)][str(i - 1)]['q']))
                    q_delta = q_t_1 * q_t_0.inv()
                    euler = q_delta.as_euler('xyz')
                    # WTF?! rotations works but i cant explain it
                    euler[1] *= -1
                    euler[2] *= -1
                    q_delta = Rotation.from_euler('xyz', euler)

                    R_delta = torch.from_numpy(q_delta.as_matrix()).type(torch.float32).to(device="cuda")
                    # R_delta = R_delta.T
                    gaussians_object_list[gs_object_id].apply_translation_on_xyz(t=t_delta)
                    gaussians_object_list[gs_object_id].apply_rotation_on_xyz(R=R_delta)
                    gaussians_object_list[gs_object_id].apply_rotation_on_splats(R=R_delta)
                    gaussians_object_list[gs_object_id].apply_rotation_on_sh(R=R_delta)

                for gs_object_id in gaussians_object_list.keys():
                    gaussian_scene.merge_gaussians(gaussian=gaussians_object_list[gs_object_id])

            i += 1

            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam is not None:
                    net_image = render(custom_cam, gaussian_scene, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2,
                                                                                                               0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)

            except Exception as e:
                network_gui.conn = None
                print("Termination error: ", e)

            if i < len(cams):
                viewpoint_cam = cams[i]

            # custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
            render_pkg = render(viewpoint_cam, gaussian_scene, pipe, background)

            net_image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg[
                "viewspace_points"], \
                render_pkg["visibility_filter"], render_pkg["radii"]

            if i > 0 and i < len(cams):
                image = (np.asarray(net_image.cpu().permute((1, 2, 0))) * 255).astype('uint8')
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                video.write(image)
            elif i > len(cams):
                cv2.destroyAllWindows()
                video.release()
                exit()

            # plt.imshow(net_image.cpu().permute((1, 2, 0)))
            # plt.show()


if __name__ == '__main__':
    # from src.dataset_objects import WoodenBlock, CupNoodle01
    # from src.dataset_envs import PlainTableSetup_01, PlainTableSetup_02, Garden

    DATASET_PATH = 'C:\\Users\\meyerls\\Documents\\AIST\\data\\pegasus_dataset'

    gaussian_splatting_manipulation(physics_file='../../engine/simulation_steps.json', dataset_path=DATASET_PATH)
