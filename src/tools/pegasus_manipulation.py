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
import os

import copy

sys.path.append("./gaussian-splatting")
sys.path.append("C:/Users/meyerls/Documents/AIST/code/gaussian-splatting/colmap-wrapper")

from scene.cameras import Camera
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from gaussian_renderer import render, network_gui
# from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary
from utils.graphics_utils import focal2fov
from utils.sh_utils import RGB2SH
from utils.general_utils import safe_state

# Own
from colmap_wrapper.dataloader import (read_images_binary, read_cameras_binary)
from src.utility.pose_interpolation import interpolate_pose
from src.utility.graphic_utils import *
import src.dataset.dataset_envs as env_assets
import src.dataset_objects as object_assets

import tqdm
import numpy as np
import json
import torch
import cv2
from pathlib import Path
from scipy.spatial.transform import Rotation


class PegasusSetup(object):
    def __init__(self, pybullet_trajectory_path, dataset_path):
        self.pybullet_trajectory_path = pybullet_trajectory_path
        self.pyhsics_data = self.load_json(file=pybullet_trajectory_path)

        self.dataset_path = dataset_path

        # Select env parameters
        environment = self.pyhsics_data['asset_infos']['environment']
        self.environment_name = list(environment.keys())[0]  # select dataset name
        self.environment_class_name = environment[self.environment_name]['class_name']
        self.environment = getattr(env_assets, self.environment_class_name)(dataset_path=dataset_path)

        self.object_data = self.pyhsics_data['asset_infos']['object']
        self.object_trajectory = self.pyhsics_data['trajectory']

        # Read data from gs
        camera_json_path = Path(self.environment.gs_model_path) / 'cameras.json'
        self.camera_data = self.load_json(camera_json_path)

        # load colmap data
        self.cam_extr = read_images_binary(Path(self.environment.reconstruction_path) / 'sparse/0/images.bin')
        self.cam_intr = read_cameras_binary(Path(self.environment.reconstruction_path) / 'sparse/0/cameras.bin')

        self.render_height = 1067
        self.render_width = 1600

        self.mode: Literal["dynamic", "static"] = 'dynamic'

    def load_json(self, file):
        with open(file) as data_file:
            data_loaded = json.load(data_file)
        return data_loaded

    def create_trajectory(self, num_cameras=5, num_interpolation_steps=24):
        cams = []

        for pose_idx in range(0, num_cameras):

            idx = sorted(self.cam_extr.keys())[pose_idx]
            idx_next = sorted(self.cam_extr.keys())[pose_idx + 1]

            pose1 = np.eye(4)
            R = np.transpose(qvec2rotmat(self.cam_extr[idx].qvec))
            t = np.array(self.cam_extr[idx].tvec)
            pose1[:3, :3] = R
            pose1[:3, 3] = t

            pose2 = np.eye(4)
            R = np.transpose(qvec2rotmat(self.cam_extr[idx_next].qvec))
            t = np.array(self.cam_extr[idx_next].tvec)
            pose2[:3, :3] = R
            pose2[:3, 3] = t

            for frame_idx in np.linspace(0, 1, num_interpolation_steps)[:-1]:
                T = interpolate_pose(t=frame_idx, t1=0, pose1=pose1, t2=1, pose2=pose2)

                # FovY = focal2fov(np.asarray(camera_data[0]['fy']), cam_intr[1].height)
                # FovX = focal2fov(np.asarray(camera_data[0]['fx']), cam_intr[1].width)

                focal_length_x = np.asarray(self.camera_data[0]['fx'])
                focal_length_y = np.asarray(self.camera_data[0]['fx'])
                FovY = focal2fov(focal_length_y, self.cam_intr[1].height)
                FovX = focal2fov(focal_length_x, self.cam_intr[1].width)

                R = T[:3, :3]
                t = np.array(T[:3, 3])

                # FovY = focal2fov(np.asarray(camera_data[0]['fy']), height)
                # FovX = focal2fov(np.asarray(camera_data[0]['fx']), width)

                viewpoint_cam = Camera(colmap_id=1,
                                       R=R,
                                       # T=np.asarray(camera_data[0]['position']),
                                       T=t,
                                       FoVx=FovX,
                                       FoVy=FovY,
                                       image=torch.empty((3, self.render_height, self.render_width)),
                                       gt_alpha_mask=None,
                                       image_name='interpolated',
                                       uid=0,
                                       data_device='cuda')
                cams.append(viewpoint_cam)

        return cams

    def load_object_gs(self, sh_degree: int, load_iteration: int = 30_000):
        gaussians_object_list = {}
        for object_name in self.object_data.keys():
            for id in self.object_data[object_name]['id']:
                obj_class_name = self.object_data[object_name]['class_name']
                obj = getattr(object_assets, obj_class_name)(dataset_path=self.dataset_path)

                obj.mode = 'fused'
                gs_object = GaussianModel(sh_degree)
                gs_object.load_ply(obj.gaussian_point_cloud_path(iteration=load_iteration))

                gaussians_object_list.update({id: gs_object})

        return gaussians_object_list

    def init_object_pose(self, gaussians_object_list: dict):
        self.mode = 'dynamic'
        for object_id in gaussians_object_list.keys():
            first_timestep_idx = 0

            t_init = -torch.asarray((self.object_trajectory[str(object_id)][str(first_timestep_idx)]['t']),
                                    dtype=torch.float32, device='cuda')

            q = Rotation.from_quat(np.asarray(self.object_trajectory[str(object_id)][str(first_timestep_idx)]['q']))
            euler = q.as_euler('xyz')
            # WTF?! rotations works but i cant explain it
            euler[1] *= -1
            euler[2] *= -1
            # euler[0] *= -1
            q_delta = Rotation.from_euler('xyz', euler)

            R_delta = torch.from_numpy(q_delta.as_matrix()).type(torch.float32).to(device="cuda")
            # R_delta = R_delta.T

            self.apply_transformation_on_gs(gs_object=gaussians_object_list[object_id], R=R_delta, t=t_init)

        return gaussians_object_list

    def update_object_pose(self, gaussians_object_list: dict, timestep: int):
        for object_id in gaussians_object_list.keys():
            t_delta = torch.from_numpy(
                np.asarray(self.object_trajectory[str(object_id)][str(timestep - 1)]['t']) - np.asarray(
                    self.object_trajectory[str(object_id)][str(timestep)]['t'])).type(
                torch.float32).to(device="cuda")
            q_t_1 = Rotation.from_quat(np.asarray(self.object_trajectory[str(object_id)][str(timestep)]['q']))
            q_t_0 = Rotation.from_quat(
                np.asarray(self.object_trajectory[str(object_id)][str(timestep - 1)]['q']))
            q_delta = q_t_1 * q_t_0.inv()

            euler = q_delta.as_euler('xyz')
            # WTF?! rotations works but i cant explain it
            euler[1] *= -1
            euler[2] *= -1
            # euler[0] *= -1
            q_delta = Rotation.from_euler('xyz', euler)

            R_delta = torch.from_numpy(q_delta.as_matrix()).type(torch.float32).to(device="cuda")
            # R_delta = R_delta.T

            self.apply_transformation_on_gs(gs_object=gaussians_object_list[object_id], R=R_delta, t=t_delta)

        return gaussians_object_list

    def apply_transformation_on_gs(self, gs_object, R, t):
        gs_object.center_position = t
        gs_object.rotation_matrix = R
        gs_object.apply_translation_on_xyz(t=t)
        gs_object.apply_rotation_on_xyz(R=R)
        gs_object.apply_rotation_on_splats(R=R)
        gs_object.apply_rotation_on_sh(R=R)

    def static_object_pose(self, gaussians_object_list: dict):
        self.mode = 'static'

        for object_id in gaussians_object_list.keys():
            last_time_step_idx = list(self.object_trajectory[str(1)].keys())[-1]

            t_init = -torch.asarray((self.object_trajectory[str(object_id)][str(last_time_step_idx)]['t']),
                                    dtype=torch.float32, device='cuda')

            q = Rotation.from_quat(np.asarray(self.object_trajectory[str(object_id)][str(last_time_step_idx)]['q']))
            euler = q.as_euler('xyz')
            # WTF?! rotations works but i cant explain it
            euler[1] *= -1
            euler[2] *= -1
            # euler[0] *= -1
            q_delta = Rotation.from_euler('xyz', euler)

            R_delta = torch.from_numpy(q_delta.as_matrix()).type(torch.float32).to(device="cuda")
            # R_delta = R_delta.T

            self.apply_transformation_on_gs(gs_object=gaussians_object_list[object_id], R=R_delta, t=t_init)

        return gaussians_object_list

    def draw_object_center(self, image, gaussians_object_list: dict, camera: Camera, semantic_colors):
        for object_name in self.object_data.keys():
            for object_id in self.object_data[object_name]['id']:
                point = torch.asarray([[0, 0, 0, 1]], dtype=torch.float32, device='cuda')
                object_offset = torch.asarray(self.object_data[object_name]['center_of_mass'],
                                              dtype=torch.float32,
                                              device='cuda')
                point[0, :3] = gaussians_object_list[object_id].center_position - object_offset

                camera_coord_hom = camera.world_view_transform.T @ point.T
                # camera_coord = geom_transform_points(view_coord, viewpoint_cam.full_proj_transform)
                # image_coord = viewpoint_cam.projection_matrix[:3, :3] @ view_coord.T
                camera_coord = camera_coord_hom.T[0] / camera_coord_hom.T[0, -1]
                camera_coord = camera_coord[:3]
                # K = torch.eye(3, dtype=torch.float32, device='cuda')
                # K[0, 0] = fov2focal(FovX, height)
                # K[1, 1] = fov2focal(FovY, width)
                # K[0, 2] = height // 2
                # K[1, 2] = width // 2

                # image_coord = K @ camera_coord
                # pixel_coord = torch.floor(image_coord / image_coord[-1]).to(torch.int).cpu().detach()
                a = camera.projection_matrix.T @ camera_coord_hom
                b = (a.T[0] / a.T[0, -1])[:3]
                b = b / b[-1]
                # ndc = (camera_coord / camera_coord[-1])
                screen_coord = ndc_to_screen(b[0], -b[1], self.render_width, self.render_height)

                color = (int(semantic_colors[object_id - 1].cpu()[0] * 255),
                         int(semantic_colors[object_id - 1].cpu()[1] * 255),
                         int(semantic_colors[object_id - 1].cpu()[2] * 255))  # BGR color
                radius = 10  # Point radius (size)
                # Draw a point on the image
                image = cv2.circle(image, (int(screen_coord[0]), int(screen_coord[1])), radius, color, -1)
        return image

    def init_video_streams(self, output='./output'):
        video_path = output
        os.makedirs(video_path, exist_ok=True)

        rgb_name = os.path.join(video_path, 'rgb_video.mp4')
        object_center_name = os.path.join(video_path, 'object_center_video.mp4')
        seg_name = os.path.join(video_path, 'seg_video.mp4')
        rgb_seg_name = os.path.join(video_path, 'rgb_seg_video.mp4')
        depth_name = os.path.join(video_path, 'depth_video.mp4')

        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        frames = 40
        image_size = (self.render_width, self.render_height)
        self.rgb_video_writer = cv2.VideoWriter(rgb_name, fourcc, frames, image_size)
        self.object_center_video_writer = cv2.VideoWriter(object_center_name, fourcc, frames, image_size)
        self.seg_video_writer = cv2.VideoWriter(seg_name, fourcc, frames, image_size)
        self.rgb_seg_video_writer = cv2.VideoWriter(rgb_seg_name, fourcc, frames, image_size)
        self.depth_video_writer = cv2.VideoWriter(depth_name, fourcc, frames, image_size)


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

    pegasus_setup = PegasusSetup(pybullet_trajectory_path=physics_file, dataset_path=dataset_path)
    viewport_cam_list = pegasus_setup.create_trajectory(num_cameras=10, num_interpolation_steps=24)

    sys.argv.append('-m')
    sys.argv.append(pegasus_setup.environment.gs_model_path)
    args = get_combined_args(parser)

    print("Rendering Environment" + pegasus_setup.environment.reconstruction_path)

    network_gui.init(IP, PORT)

    # Initialize system state (RNG)
    safe_state(QUIET)
    pipe = pipeline.extract(args)
    # render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)

    dataset = model.extract(args)

    pegasus_setup.init_video_streams(output='./output')
    with (torch.no_grad()):
        gaussian_environment = GaussianModel(dataset.sh_degree)
        gaussian_environment.load_ply(pegasus_setup.environment.gaussian_point_cloud_path(iteration=load_iteration))

        # scene = Scene(dataset, gaussian_environment, load_iteration=load_iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        gaussians_object_list = pegasus_setup.load_object_gs(sh_degree=dataset.sh_degree, load_iteration=load_iteration)

        # Static
        gaussians_object_list = pegasus_setup.static_object_pose(gaussians_object_list=gaussians_object_list)
        # Dynamic
        # gaussians_object_list = pegasus_setup.init_object_pose(gaussians_object_list=gaussians_object_list)

        semantic_colors = generate_colors(len(gaussians_object_list))

        if network_gui.conn is None:
            network_gui.try_connect()

        num_timesteps = len(viewport_cam_list)
        bar = tqdm.tqdm(total=num_timesteps)
        for i in range(num_timesteps):
            gaussian_scene = copy.deepcopy(gaussian_environment)

            # Compose scene for rgb rendering
            for gs_object_id in gaussians_object_list.keys():
                gaussian_scene.merge_gaussians(gaussian=gaussians_object_list[gs_object_id])

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
                # print("Termination error: ", e)

            if i < len(viewport_cam_list):
                viewpoint_cam = viewport_cam_list[i]

            # Render rgb
            render_pkg = render(viewpoint_cam, gaussian_scene, pipe, background)
            net_image, depth_image = render_pkg["render"], render_pkg["depth"]

            semantic_gaussians_object_list = copy.deepcopy(gaussians_object_list)
            gaussian_scene = copy.deepcopy(gaussian_environment)

            # Compose scene for segmentation rendering and set splat color according to the assigned segmentation color
            for gs_object_id in semantic_gaussians_object_list.keys():
                object_semantic_color = semantic_colors[gs_object_id - 1]
                base_color = RGB2SH(object_semantic_color)
                semantic_gaussians_object_list[gs_object_id]._features_dc[:] = base_color
                semantic_gaussians_object_list[gs_object_id]._features_rest[:, :] = torch.asarray([0, 0, 0])
                gaussian_scene.merge_gaussians(gaussian=semantic_gaussians_object_list[gs_object_id])

            # Segmentation
            mask_env = torch.ones(gaussian_scene._xyz.shape[0], dtype=bool).to('cuda')
            mask_env[:gaussian_environment._xyz.shape[0]] = False
            gaussian_scene.mask_points(mask_env)

            # Render segmentation
            render_pkg = render(viewpoint_cam, gaussian_scene, pipe, background)
            net_seg_image = render_pkg["render"]

            if i < len(viewport_cam_list):
                rgb_image = (np.ascontiguousarray(net_image.cpu().permute((1, 2, 0))) * 255).astype('uint8')
                seg_image = (np.ascontiguousarray(net_seg_image.cpu().permute((1, 2, 0))) * 255).astype('uint8')

                rgb_mask_overlay = cv2.addWeighted(rgb_image, 1, seg_image, 0.5, 0)

                object_center_image = copy.deepcopy(rgb_image)
                object_center_image = pegasus_setup.draw_object_center(image=object_center_image,
                                                                       gaussians_object_list=gaussians_object_list,
                                                                       camera=viewpoint_cam,
                                                                       semantic_colors=semantic_colors)

                rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
                seg_image = cv2.cvtColor(seg_image, cv2.COLOR_BGR2RGB)
                object_center_image = cv2.cvtColor(object_center_image, cv2.COLOR_BGR2RGB)
                rgb_mask_overlay = cv2.cvtColor(rgb_mask_overlay, cv2.COLOR_BGR2RGB)
                max_distance_in_meter = 5
                depth_image_normalized = torch.floor((depth_image / max_distance_in_meter) * 255)[
                    0].cpu().numpy().astype('uint8')
                depth_image_normalized = cv2.cvtColor(depth_image_normalized, cv2.COLOR_GRAY2BGR)

                # Save images to video stream
                pegasus_setup.rgb_video_writer.write(rgb_image)
                pegasus_setup.object_center_video_writer.write(object_center_image)
                pegasus_setup.seg_video_writer.write(seg_image)
                pegasus_setup.rgb_seg_video_writer.write(rgb_mask_overlay)
                pegasus_setup.depth_video_writer.write(depth_image_normalized)
            else:
                pegasus_setup.rgb_video_writer.release()
                pegasus_setup.object_center_video_writer.release()
                pegasus_setup.seg_video_writer.release()
                pegasus_setup.rgb_seg_video_writer.release()
                pegasus_setup.depth_video_writer.release()
                #exit()

            bar.update(1)

            if pegasus_setup.mode == 'dynamic':
                gaussians_object_list = pegasus_setup.update_object_pose(gaussians_object_list=gaussians_object_list,
                                                                         timestep=i + 1)


if __name__ == '__main__':
    DATASET_PATH = 'C:\\Users\\meyerls\\Documents\\AIST\\data\\pegasus_dataset'
    gaussian_splatting_manipulation(physics_file='src/engine/simulation_steps.json', dataset_path=DATASET_PATH)
