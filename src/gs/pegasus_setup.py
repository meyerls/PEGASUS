import sys
import os
from typing import Literal

import copy

sys.path.append("./gaussian-splatting")
sys.path.append("C:/Users/meyerls/Documents/AIST/code/gaussian-splatting/colmap-wrapper")

from scene.cameras import Camera
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
# from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary
from utils.graphics_utils import focal2fov, getWorld2View2, getWorld2View
from utils.general_utils import safe_state
import open3d as o3d
import imageio
# Own
from colmap_wrapper.dataloader import (write_images_binary, write_points3D_binary, write_cameras_binary,
                                       write_cameras_text, write_images_text, write_points3D_text,
                                       read_images_binary, read_points3d_binary, read_cameras_binary)
from src.utility.pose_interpolation import interpolate_pose
from src.utility.graphic_utils import *
# import src.dataset_envs_old as env_assets
import src.dataset.dataset_envs as env_assets
# import src.dataset_objects_old as object_assets
import src.dataset.ycb_objects as object_assets
from src.gs.gaussian_model import GaussianModel

import tqdm
import numpy as np
import json
import pylab as plt
import torch
import cv2
from pathlib import Path
from scipy.spatial.transform import Rotation


class PegasusSetup(object):
    def __init__(self,
                 pybullet_trajectory_path,
                 dataset_path,
                 render_height: int,
                 render_width: int,
                 env_dataset_path=None,
                 mode: Literal["dynamic", "static"] = 'static'):
        self.pybullet_trajectory_path = pybullet_trajectory_path
        self.pyhsics_data = self.load_json(file=pybullet_trajectory_path)

        self.dataset_path = dataset_path

        if env_dataset_path:
            self.env_dataset_path = env_dataset_path
        else:
            self.env_dataset_path = dataset_path

        # Select env parameters
        environment = self.pyhsics_data['asset_infos']['environment']
        self.environment_name = list(environment.keys())[0]  # select dataset name
        self.environment_class_name = environment[self.environment_name]['class_name']
        self.environment = getattr(env_assets, self.environment_class_name)(dataset_path=self.env_dataset_path)

        self.object_data = self.pyhsics_data['asset_infos']['object']
        self.object_trajectory = self.pyhsics_data['trajectory']

        # Read data from gs
        camera_json_path = Path(self.environment.gs_model_path) / 'cameras.json'
        self.camera_data = self.load_json(camera_json_path)

        # load colmap data ToDo this takes some time!
        # self.cam_extr = read_images_binary(Path(self.environment.reconstruction_path) / 'sparse/0/images.bin')
        # self.cam_intr = read_cameras_binary(Path(self.environment.reconstruction_path) / 'sparse/0/cameras.bin')

        self.render_height = render_height  # 1067
        self.render_width = render_width  # 1600

        self.mode: Literal["dynamic", "static"] = mode

    def load_json(self, file):
        with open(file) as data_file:
            data_loaded = json.load(data_file)
        return data_loaded

    def create_camera_trajectory(self, num_cameras=5, num_interpolation_steps=24,
                                 mode: Literal["random", "sequence", 'random+zoom'] = "random"):
        cams = []

        start_frame = np.random.randint(0, self.cam_extr.keys().__len__() - num_cameras)

        for pose_idx in range(start_frame, start_frame + num_cameras):

            idx = sorted(self.cam_extr.keys())[pose_idx]
            idx_next = sorted(self.cam_extr.keys())[pose_idx + 1]

            pose1 = np.eye(4)
            R = np.transpose(qvec2rotmat(self.cam_extr[idx].qvec))
            t = np.array(self.cam_extr[idx].tvec)
            pose1[:3, :3] = R
            pose1[:3, 3] = t
            if mode == 'random+zoom':
                # pose1[:3, 3] *= np.random.uniform(1.5, 2)
                pose1[:3, 3] *= np.random.uniform(0.6, 1)

            pose2 = np.eye(4)
            R = np.transpose(qvec2rotmat(self.cam_extr[idx_next].qvec))
            t = np.array(self.cam_extr[idx_next].tvec)
            pose2[:3, :3] = R
            pose2[:3, 3] = t  # * np.random.uniform(0.6, 1)
            if mode == 'random+zoom':
                pose1[:3, 3] *= np.random.uniform(0.6, 1)

            for frame_idx in np.linspace(0, 1, num_interpolation_steps + 1)[:-1]:
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
            for id in self.object_data[object_name]['bullet_id']:
                obj_class_name = self.object_data[object_name]['class_name']
                obj = getattr(object_assets, obj_class_name)(dataset_path=self.dataset_path)

                obj.mode = 'fused'
                gs_object = GaussianModel(sh_degree)
                gs_object.load_ply(obj.gaussian_point_cloud_path(iteration=load_iteration))
                gs_object.meta_info = obj
                gaussians_object_list.update({id: gs_object})

        return gaussians_object_list

    def dynamic_object_pose(self, gaussians_object_list: dict):
        self.mode = 'dynamic'
        for object_id in gaussians_object_list.keys():
            first_timestep_idx = 0

            q = self.object_trajectory[str(object_id)][str(first_timestep_idx)]['q']
            t = self.object_trajectory[str(object_id)][str(first_timestep_idx)]['t']
            t_init = torch.asarray(t, dtype=torch.float32, device='cuda')
            q_init = Rotation.from_quat(np.asarray(q))
            R_init = torch.from_numpy(q_init.as_matrix()).type(torch.float32).to(device="cuda")

            gaussians_object_list[object_id].R_init = R_init
            gaussians_object_list[object_id].t_init = t_init

            self.apply_transformation_on_gs(gs_object=gaussians_object_list[object_id], R=R_init, t=t_init)

        return gaussians_object_list

    def update_object_pose(self, gaussians_object_list: dict, timestep: int):
        for object_id in gaussians_object_list.keys():
            curr_step = np.asarray(self.object_trajectory[str(object_id)][str(timestep)]['t'])
            past_step = np.asarray(self.object_trajectory[str(object_id)][str(timestep - 1)]['t'])

            t_delta = torch.from_numpy(curr_step - past_step).type(torch.float32).to(device="cuda")
            q_t_1 = Rotation.from_quat(np.asarray(self.object_trajectory[str(object_id)][str(timestep)]['q']))
            q_t_0 = Rotation.from_quat(
                np.asarray(self.object_trajectory[str(object_id)][str(timestep - 1)]['q']))
            q_delta = q_t_1 * q_t_0.inv()

            R = torch.from_numpy(q_delta.as_matrix()).type(torch.float32).to(device="cuda")

            self.apply_transformation_on_gs(gs_object=gaussians_object_list[object_id], R=R, t=t_delta)

        return gaussians_object_list

    def apply_transformation_on_gs(self, gs_object, R, t):
        T = torch.eye(4, dtype=torch.float32, device='cuda')
        T[:3, :3] = R
        T[:3, 3] = t

        gs_object.center_position = t  # Just for static case
        gs_object.rotation_matrix = R
        gs_object.transformation_matrix = T
        # gs_object.apply_translation_on_xyz(t=t)
        # gs_object.apply_rotation_on_xyz(R=R)
        gs_object.apply_transformation_on_xyz(T=T)
        gs_object.apply_rotation_on_splats(R=R)
        gs_object.apply_rotation_on_sh(R=R)

    def static_object_pose(self, gaussians_object_list: dict):
        self.mode = 'static'

        for object_id in gaussians_object_list.keys():
            last_time_step_idx = list(self.object_trajectory[str(1)].keys())[-1]

            q = self.object_trajectory[str(object_id)][str(last_time_step_idx)]['q']
            t = self.object_trajectory[str(object_id)][str(last_time_step_idx)]['t']
            t_init = torch.asarray(t, dtype=torch.float32, device='cuda')
            q_init = Rotation.from_quat(np.asarray(q))
            R_init = torch.from_numpy(q_init.as_matrix()).type(torch.float32).to(device="cuda")

            gaussians_object_list[object_id].R_init = R_init
            gaussians_object_list[object_id].t_init = t_init

            self.apply_transformation_on_gs(gs_object=gaussians_object_list[object_id], R=R_init, t=t_init)

        return gaussians_object_list

    def draw_object_center(self, image, gaussians_object_list: dict, camera: Camera, semantic_colors, K):
        image = copy.deepcopy(image)

        for object_name in self.object_data.keys():
            for object_id in self.object_data[object_name]['bullet_id']:
                x = gaussians_object_list[object_id].get_point_cloud().get_center()[0]
                y = gaussians_object_list[object_id].get_point_cloud().get_center()[1]
                z = gaussians_object_list[object_id].get_point_cloud().get_center()[2]
                point = np.asarray([[x, y, z, 1]])
                # gaussians_object_list[object_id].center_position.cpu().numpy()  # - object_offset

                T_m2w = np.eye(4)
                # T_m2w[:3, :3] = gaussians_object_list[object_id].R_init.cpu().numpy()
                # T_m2w[:3, -1] = gaussians_object_list[object_id].t_init.cpu().numpy()  # - mesh.get_center()

                T_w2c = np.eye(4)
                T_w2c[:3, :3] = camera.R.T
                T_w2c[:3, 3] = camera.T

                T = (T_w2c @ T_m2w)

                P = K @ T[:3]
                ret = P @ point.T

                c = cv2.convertPointsFromHomogeneous(ret.T)[:, 0, :].astype(int)

                color = (int(semantic_colors[object_id - 1].cpu()[0] * 255),
                         int(semantic_colors[object_id - 1].cpu()[1] * 255),
                         int(semantic_colors[object_id - 1].cpu()[2] * 255))  # BGR color
                radius = 6  # Point radius (size)
                # Draw a point on the image
                image = cv2.circle(image, (int(c[0, 0]), int(c[0, 1])), radius, color, -1)
        return image

    def init_video_streams(self, output='./output', fps: int = 10):
        video_path = output
        os.makedirs(video_path, exist_ok=True)

        rgb_name = os.path.join(video_path, 'rgb_video.mp4')
        object_center_name = os.path.join(video_path, 'object_center_video.mp4')
        seg_name = os.path.join(video_path, 'seg_video.mp4')
        rgb_seg_name = os.path.join(video_path, 'rgb_seg_video.mp4')
        depth_name = os.path.join(video_path, 'depth_video.mp4')

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        image_size = (self.render_width, self.render_height)
        self.rgb_video_writer = cv2.VideoWriter(rgb_name, fourcc, fps, image_size)
        self.object_center_video_writer = cv2.VideoWriter(object_center_name, fourcc, fps, image_size)
        self.seg_video_writer = cv2.VideoWriter(seg_name, fourcc, fps, image_size)
        self.rgb_seg_video_writer = cv2.VideoWriter(rgb_seg_name, fourcc, fps, image_size)
        self.depth_video_writer = cv2.VideoWriter(depth_name, fourcc, fps, image_size)

    def close_video_streams(self):
        self.rgb_video_writer.release()
        self.object_center_video_writer.release()
        self.seg_video_writer.release()
        self.rgb_seg_video_writer.release()
        self.depth_video_writer.release()

    def write_image2video(self, rgb, depth, seg, center_image, max_distance_in_meter=5):

        seg_image = None
        if isinstance(seg, torch.Tensor):
            seg_image = (np.ascontiguousarray(seg.cpu()) * 255).astype('uint8')
            rgb_mask_overlay = cv2.addWeighted(rgb, 1, seg_image, 0.5, 0)

        # Save images to video stream
        if isinstance(rgb, np.ndarray):
            self.rgb_video_writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if isinstance(center_image, np.ndarray):
            self.object_center_video_writer.write(cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB))
        if isinstance(seg_image, np.ndarray):
            self.seg_video_writer.write(cv2.cvtColor(seg_image, cv2.COLOR_BGR2RGB))
            self.rgb_seg_video_writer.write(cv2.cvtColor(rgb_mask_overlay, cv2.COLOR_BGR2RGB))
        if isinstance(depth, torch.Tensor):
            depth_image_normalized = torch.floor((depth / max_distance_in_meter) * 255)[..., 0].cpu().numpy().astype(
                'uint8')
            depth_image_normalized = cv2.cvtColor(depth_image_normalized, cv2.COLOR_GRAY2BGR)
            self.depth_video_writer.write(depth_image_normalized)
