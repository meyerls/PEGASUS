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
from typing import Literal

import copy

sys.path.append("./gaussian-splatting")
sys.path.append("C:/Users/meyerls/Documents/AIST/code/gaussian-splatting/colmap-wrapper")

from scene import Scene
from scene.cameras import Camera
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render, network_gui
# from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary
from utils.graphics_utils import focal2fov, getWorld2View2, getWorld2View, geom_transform_points, fov2focal
from utils.sh_utils import RGB2SH, SH2RGB
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
import src.dataset_objects as object_assets
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
    def __init__(self, pybullet_trajectory_path, dataset_path, render_height: int, render_width: int):
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

        self.render_height = render_height  # 1067
        self.render_width = render_width  # 1600

        self.mode: Literal["dynamic", "static"] = 'static'

    def load_json(self, file):
        with open(file) as data_file:
            data_loaded = json.load(data_file)
        return data_loaded

    def create_camera_trajectory(self, num_cameras=5, num_interpolation_steps=24):
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
        return NotImplemented
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
    def init_object_pose(self, gaussians_object_list: dict):
        return NotImplemented
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
            t_delta = torch.from_numpy(
                np.asarray(self.object_trajectory[str(object_id)][str(timestep - 1)]['t']) - np.asarray(
                    self.object_trajectory[str(object_id)][str(timestep)]['t'])).type(
                torch.float32).to(device="cuda")
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

        gs_object.center_position = t
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
                point = np.asarray([[0, 0, 0, 1]])
                point[0, :3] = gaussians_object_list[object_id].center_position.cpu().numpy()  # - object_offset

                T_m2w = np.eye(4)
                T_m2w[:3, :3] = gaussians_object_list[object_id].R_init.cpu().numpy()
                T_m2w[:3, -1] = gaussians_object_list[object_id].t_init.cpu().numpy()  # - mesh.get_center()

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

        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
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


class PegasusBOPDatasetWriter:
    def __init__(self, dataset_name: str,
                 dataset_output_path: Literal[str, Path],
                 camera_intr,
                 render_width: int,
                 render_height: int,
                 object_models: dict,
                 object_dataset_path: str,
                 scene_id: int):
        self.dataset_name: str = dataset_name
        self.dataset_output_basepath: Literal[str, Path] = dataset_output_path

        self.dataset_path = self.dataset_output_basepath / self.dataset_name
        self.dataset_path.mkdir(parents=True, exist_ok=True)

        self.camera_intr = camera_intr
        self.render_width = render_width
        self.render_height = render_height

        self.object_models = object_models
        self.object_dataset_path = object_dataset_path

        self.model_path = self.dataset_path / 'models'
        self.model_path.mkdir(parents=True, exist_ok=True)

        self.write_camera_json(file_name='camera.json')
        self.write_models(model_path=self.model_path, model_info_json_file_name='models_info.json')

        self.train_data_path = self.dataset_path / 'train'
        self.scene_path = self.train_data_path / "{:06d}".format(scene_id)

        self.depth_path = self.scene_path / 'depth'
        self.depth_path.mkdir(parents=True, exist_ok=True)
        self.mask_visib_path = self.scene_path / 'mask_visib'
        self.mask_visib_path.mkdir(parents=True, exist_ok=True)
        self.mask_path = self.scene_path / 'mask'
        self.mask_path.mkdir(parents=True, exist_ok=True)
        self.rgb_path = self.scene_path / 'rgb'
        self.rgb_path.mkdir(parents=True, exist_ok=True)

        self.scene_camera_json_path = self.scene_path / 'scene_camera.json'
        self.scene_camera_json = {}

        self.scene_gt_json_path = self.scene_path / 'scene_gt.json'
        self.scene_gt_json = {}

        self.scene_id = scene_id

    def write_camera_json(self, file_name: str):
        focal_length_x = np.asarray(self.camera_intr[1].params[0])
        focal_length_y = np.asarray(self.camera_intr[1].params[1])

        FovX = focal2fov(focal_length_x, self.camera_intr[1].width)
        FovY = focal2fov(focal_length_y, self.camera_intr[1].height)

        fx = fov2focal(fov=FovX, pixels=self.render_width)
        fy = fov2focal(fov=FovY, pixels=self.render_height)

        self.camera_json = {
            'cx': self.render_width / 2,
            'cy': self.render_height / 2,
            'depth_scale': 1.0,  # for millimeter # it is true to scale
            'fx': fx,
            'fy': fy,
            'height': self.render_height,
            'width': self.render_width,
        }

        with open(Path(self.dataset_path / file_name), 'w') as f:
            json.dump(self.camera_json, f, indent=4)

    @staticmethod
    def calculate_diameter_of_object(obj):
        # Get the vertices from the mesh
        vertices = np.asarray(obj.vertices)

        # Calculate the diameter
        max_distance = 0

        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                distance = np.linalg.norm(vertices[i] - vertices[j])
                max_distance = max(max_distance, distance)
        return max_distance

    def write_models(self, model_path: Literal[Path, str], model_info_json_file_name: str):
        self.model_info_json = {}
        for obj_id, object_name in enumerate(self.object_models):
            object_mesh_path = Path(self.object_dataset_path) / 'urdf' / (object_name + '.obj')
            obj_mesh = o3d.io.read_triangle_mesh(object_mesh_path.__str__())
            axis_aligned_bbox = obj_mesh.get_axis_aligned_bounding_box()

            self.model_info_json.update(
                {str(obj_id + 1): {"diameter": self.calculate_diameter_of_object(obj=obj_mesh),
                                   "min_x": axis_aligned_bbox.min_bound[0],
                                   "min_y": axis_aligned_bbox.min_bound[1],
                                   "min_z": axis_aligned_bbox.min_bound[2],
                                   "size_x": axis_aligned_bbox.get_extent()[0],
                                   "size_y": axis_aligned_bbox.get_extent()[1],
                                   "size_z": axis_aligned_bbox.get_extent()[2]}})

            ply_output_path = model_path / 'obj_{:06d}.ply'.format(obj_id + 1)
            o3d.io.write_triangle_mesh(ply_output_path.__str__(), obj_mesh, write_ascii=True)

            if False:
                bb = obj_mesh.get_axis_aligned_bounding_box()
                bb.color = (0.5, 0.5, 0.5)
                o3d.visualization.draw_geometries([obj_mesh, bb])

        with open(Path(model_path / model_info_json_file_name), 'w') as f:
            json.dump(self.model_info_json, f, indent=1)

    def write_training_data(self, rgb_image: np.ndarray,
                            seg_image: np.ndarray,
                            mask_silhouette: np.ndarray,
                            depth_image: np.ndarray,
                            frame_id: int):

        if isinstance(rgb_image, np.ndarray):
            rgb_file_name = self.rgb_path / "{:06d}.png".format(frame_id)
            imageio.imwrite(rgb_file_name, rgb_image)

        if isinstance(depth_image, np.ndarray):
            depth_file_name = self.depth_path / "{:06d}.png".format(frame_id)
            imageio.imwrite(depth_file_name, depth_image[..., 0])

        if isinstance(mask_silhouette, np.ndarray):
            for obj_id in range(mask_silhouette.shape[-1]):
                mask_file_name = self.mask_path / "{:06d}_{:06d}.png".format(frame_id, obj_id)
                imageio.imwrite(mask_file_name, mask_silhouette[..., obj_id].astype(np.uint8) * 255)

        if isinstance(seg_image, np.ndarray):
            for obj_id in range(seg_image.shape[-1]):
                mask_vis_file_name = self.mask_visib_path / "{:06d}_{:06d}.png".format(frame_id, obj_id)
                imageio.imwrite(mask_vis_file_name, seg_image[..., obj_id].astype(np.uint8) * 255)

    def add_scene_camera_json(self, frame_id: int):
        K = np.eye(3, dtype=np.float64)
        K[0, 0] = self.camera_json['fx']
        K[1, 1] = self.camera_json['fy']
        K[0, 2] = self.camera_json['cx']
        K[1, 2] = self.camera_json['cy']

        self.scene_camera_json.update(
            {
                frame_id: {
                    'cam_K': list(K.flatten()),
                    'depth_scale': 1.0
                }
            })
        self.K = K

    def add_scene_gt_json(self, time_step, gs_object_list, cam, rgb_image, debug=False):

        # Add timestep if not existing
        if not str(time_step) in self.scene_gt_json.keys():
            self.scene_gt_json.update({str(time_step): []})

        # Compute world to camera system transformation
        T_w2c = np.eye(4)
        T_w2c[:3, :3] = cam.R.T
        T_w2c[:3, 3] = cam.T

        for object_id in gs_object_list.keys():
            mesh = o3d.io.read_triangle_mesh(gs_object_list[object_id].meta_info.urdf_obj_path)
            mesh_bb = mesh.get_minimal_oriented_bounding_box(robust=True)
            mesh_bb_np = np.asarray(mesh_bb.get_box_points())
            # change of order arcording to open3d and ndds
            #
            #              4 +-----------------+ 1                              (m) 3 +-----------------+ 0 (b)
            #               /                 /|                                     /                 /|
            #              /                 / |                                    /                 / |
            #           6 +-----------------+ 3|                             (m) 2 +-----------------+ 1| (b)
            #             |                 |  |                                   |                 |  |
            #             |       ^ z       |  |                                   |       ^ z       |  |
            #             |       |         |  |              ---->                |       |         |  |
            #             |       x --> y   |  |                                   |  y <--x         |  |
            #             |     7+          |  + 2                             (y) |                 |  + 4 (g)
            #             |                 | /                                    |                 | /
            #             |                 |/                                     |                 |/
            #           5 +-----------------+ 8                              (y) 6 +-----------------+ 5 (g)
            #

            mesh_bb_np = np.vstack([mesh_bb_np[0],
                                    mesh_bb_np[2],
                                    mesh_bb_np[5],
                                    mesh_bb_np[3],
                                    mesh_bb_np[1],
                                    mesh_bb_np[7],
                                    mesh_bb_np[4],
                                    mesh_bb_np[6]])

            # Model to World transformation
            T_m2w = np.eye(4)
            T_m2w[:3, :3] = gs_object_list[object_id].R_init.cpu().numpy()
            T_m2w[:3, -1] = gs_object_list[object_id].t_init.cpu().numpy()  # - mesh.get_center()

            T = T_w2c @ T_m2w

            if debug:
                mesh_bb.color = (0, 0, 0)
                o3d.visualization.draw_geometries([mesh_bb, mesh])

                points = np.ones((8, 4))
                points[:, :3] = np.asarray(mesh_bb_np)

                P = self.K @ T[:3]
                ret = P @ points.T

                c = cv2.convertPointsFromHomogeneous(ret.T)[:, 0, :].astype(int)

                for point_idx in range(c.shape[0]):
                    # print(c[point_idx])
                    rgb_image = cv2.circle(rgb_image, (int(c[point_idx, 0]), int(c[point_idx, 1])), 3,
                                           (128, 0, 0),
                                           -1)

                plt.imshow(rgb_image)
                plt.show()

            if debug:  # debug
                mesh = o3d.io.read_triangle_mesh(gs_object_list[object_id].meta_info.urdf_obj_path)
                # mesh = mesh.translate(mesh.get_center())
                coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                # o3d.visualization.draw_geometries([mesh, coord])

                points = np.ones((100, 4))
                points[:, :3] = np.asarray(mesh.sample_points_uniformly(points.shape[0]).points)
                # points[:, :3] = gaussians_object_list[1]._xyz[:points.shape[0]].cpu().numpy()

                P = self.K @ T[:3]
                ret = P @ points.T

                c = cv2.convertPointsFromHomogeneous(ret.T)[:, 0, :].astype(int)

                for point_idx in range(c.shape[0]):
                    # print(c[point_idx])
                    rgb_image = cv2.circle(rgb_image, (int(c[point_idx, 0]), int(c[point_idx, 1])), 3,
                                           (128, 0, 0),
                                           -1)

            R_m2c = T[:3, :3]
            t_m2c = T[:3, 3]

            # Compute projected 3d bounding box in image coord
            bb3d_points = np.ones((8, 4))
            bb3d_points[:, :3] = np.asarray(mesh_bb_np)

            P = self.K @ T[:3]
            project_points_hom = P @ bb3d_points.T
            project_points = cv2.convertPointsFromHomogeneous(project_points_hom.T)[:, 0, :]

            # Compute projected 3d bounding box CENTER in image coord
            bb3d_center_point = np.ones((1, 4))
            bb3d_center_point[:, :3] = mesh_bb.get_center()

            P = self.K @ T[:3]
            project_center_point_hom = P @ bb3d_center_point.T
            project_center_point = cv2.convertPointsFromHomogeneous(project_center_point_hom.T)[:, 0, :]

            self.scene_gt_json[str(time_step)].append({
                'cam_R_m2c': list(R_m2c.flatten()),
                'cam_t_m2c': list(t_m2c.flatten()),
                'T_w2c': list(T_w2c.flatten()),
                'T_m2w': list(T_m2w.flatten()),
                'obj_id': object_id,
                '3d_bounding_box_model_coord': mesh_bb_np.tolist(),
                '3d_bounding_center': mesh.get_center().tolist(),
                'projected_center': project_center_point.tolist(),
                'projected_points': project_points.tolist(),
            })

        if debug:
            plt.imshow(rgb_image)
            plt.show()

    def write_scene_camera_json(self):
        with open(self.scene_camera_json_path, 'w') as f:
            json.dump(self.scene_camera_json, f, indent=1)

    def write_scene_gt_json(self):
        with open(self.scene_gt_json_path, 'w') as f:
            json.dump(self.scene_gt_json, f, indent=1)

    def write_targets_bop19(self):

        return NotImplemented


def render_rgb_and_depth(cam, gs_scene, pipe_settings, bg, debug=False):
    # Render rgb
    render_pkg = render(cam, gs_scene, pipe_settings, bg)
    net_image, depth_image = render_pkg["render"], render_pkg["depth"]

    rgb_image = net_image.cpu().permute((1, 2, 0))
    depth_image = depth_image.cpu().permute((1, 2, 0))

    if debug:
        depth_image2save = np.asarray((depth_image / depth_image.max() * 255).cpu()).astype('uint8')
        imageio.imwrite('image_depth.png', depth_image2save[..., 0])
        plt.imshow(depth_image2save)
        plt.show()

        rgb_image2save = np.asarray((rgb_image * 255).cpu()).astype('uint8')
        imageio.imwrite('image_rgb.png', rgb_image2save)
        plt.imshow(rgb_image2save)
        plt.show()

    return rgb_image, depth_image


def render_silhouette_mask(cam, gs_object_list, gs_env, width, height, color_set, pipe_settings, bg):
    semantic_gaussians_object_list = copy.deepcopy(gs_object_list)

    gaussian_scene_black = copy.deepcopy(gs_env)
    mask_env = torch.ones(gaussian_scene_black._xyz.shape[0], dtype=bool).to('cuda')
    mask_env[:gs_env._xyz.shape[0]] = False
    gaussian_scene_black.mask_points(mask_env)

    mask_silhouette = np.zeros((height, width, color_set.shape[0]))
    bb_list = []
    # Compose scene for segmentation rendering and set splat color according to the assigned segmentation color
    for gs_object_id in semantic_gaussians_object_list.keys():
        gaussian_scene_black_temp = copy.deepcopy(gaussian_scene_black)
        object_semantic_color = color_set[gs_object_id - 1]
        base_color = RGB2SH(object_semantic_color)
        semantic_gaussians_object_list[gs_object_id]._features_dc[:] = base_color
        semantic_gaussians_object_list[gs_object_id]._features_rest[:, :] = torch.asarray([0, 0, 0])

        # pcd = semantic_gaussians_object_list[gs_object_id].get_point_cloud()
        # bb = pcd.get_minimal_oriented_bounding_box()
        # if False:
        #    bb.color = (0, 0, 0)
        #    o3d.visualization.draw_geometries([bb, pcd])
        # bb_array = np.asarray(bb.get_box_points())
        # bb_list.append(bb_array)
        gaussian_scene_black_temp.merge_gaussians(gaussian=semantic_gaussians_object_list[gs_object_id])

        # Render segmentation
        render_pkg = render(cam, gaussian_scene_black_temp, pipe_settings, bg)
        net_seg_image_silhoutte = render_pkg["render"]

        seg_silhouette_mask = net_seg_image_silhoutte.cpu().permute((1, 2, 0))

        distance = np.linalg.norm(seg_silhouette_mask.detach().cpu() - object_semantic_color.cpu().numpy(), axis=2)
        mask_silhouette[distance <= 0.1, gs_object_id - 1] = 1

    return mask_silhouette  # , bb_list


def render_visib_mask(cam, gs_environment, gs_object_list, color_set, height, width, pipe_settings, bg):
    gaussian_scene = copy.deepcopy(gs_environment)
    semantic_gaussians_object_list = copy.deepcopy(gs_object_list)

    # Compose scene for segmentation rendering and set splat color according to the assigned segmentation color
    for gs_object_id in semantic_gaussians_object_list.keys():
        object_semantic_color = color_set[gs_object_id - 1]
        base_color = RGB2SH(object_semantic_color)
        semantic_gaussians_object_list[gs_object_id]._features_dc[:] = base_color
        semantic_gaussians_object_list[gs_object_id]._features_rest[:, :] = torch.asarray([0, 0, 0])
        gaussian_scene.merge_gaussians(gaussian=semantic_gaussians_object_list[gs_object_id])

    # Segmentation
    mask_env = torch.ones(gaussian_scene._xyz.shape[0], dtype=bool).to('cuda')
    mask_env[:gs_environment._xyz.shape[0]] = False
    gaussian_scene.mask_points(mask_env)

    # Render segmentation
    render_pkg = render(cam, gaussian_scene, pipe_settings, bg)
    net_seg_image = render_pkg["render"]

    seg_mask = net_seg_image.cpu().permute((1, 2, 0))
    invidiual_seg_masks = np.zeros((height, width, color_set.shape[0]))
    for c_i, c in enumerate(color_set):
        distance = np.linalg.norm(seg_mask - c.cpu().numpy(), axis=2)
        invidiual_seg_masks[distance <= 0.1, c_i] = 1  # to do change value!!

    seg_image = net_seg_image.cpu().permute((1, 2, 0))

    return invidiual_seg_masks, seg_image


def render_semanticsegmentation_mask(cam, gs_environment, gs_object_list, color_set, height, width, pipe_settings, bg,
                                     debug):
    gaussian_scene = copy.deepcopy(gs_environment)
    semantic_gaussians_object_list = copy.deepcopy(gs_object_list)

    # Compose scene for segmentation rendering and set splat color according to the assigned segmentation color
    for gs_object_id in semantic_gaussians_object_list.keys():
        object_semantic_color = color_set[gs_object_id - 1]
        base_color = RGB2SH(object_semantic_color)
        semantic_gaussians_object_list[gs_object_id]._features_dc[:] = base_color
        semantic_gaussians_object_list[gs_object_id]._features_rest[:, :] = torch.asarray([0, 0, 0])
        gaussian_scene.merge_gaussians(gaussian=semantic_gaussians_object_list[gs_object_id])

    # Segmentation
    mask_env = torch.ones(gaussian_scene._xyz.shape[0], dtype=bool).to('cuda')
    mask_env[:gs_environment._xyz.shape[0]] = False
    gaussian_scene.mask_points(mask_env)

    # Render segmentation
    render_pkg = render(cam, gaussian_scene, pipe_settings, bg)
    net_seg_image = render_pkg["render"]

    seg_mask = net_seg_image.cpu()

    if debug:
        semantic_mask_image2save = (
                np.ascontiguousarray(seg_mask.permute((1, 2, 0))) * 255).astype(
            'uint8')
        cv2.imwrite('image_semantic_mask.png', semantic_mask_image2save)

    return seg_mask


def gaussian_splatting_manipulation(physics_file: str,
                                    dataset_path: str,
                                    data_points: list = [],
                                    render_height: int = 480,
                                    render_width: int = 640,
                                    save_bop: bool = True,
                                    scene_id: int = 1,
                                    load_iteration: int = 30_000,
                                    IP: str = "127.0.0.1",
                                    PORT: int = 6009,
                                    QUIET: bool = False):
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    pegasus_setup = PegasusSetup(pybullet_trajectory_path=physics_file,
                                 dataset_path=dataset_path,
                                 render_height=render_height,
                                 render_width=render_width)
    pegasus_dataset = PegasusBOPDatasetWriter(dataset_name='pegasus',
                                              dataset_output_path=Path('./dataset'),
                                              camera_intr=pegasus_setup.cam_intr,
                                              render_width=pegasus_setup.render_width,
                                              render_height=pegasus_setup.render_height,
                                              object_models=pegasus_setup.object_data.keys(),
                                              object_dataset_path=dataset_path,
                                              scene_id=scene_id)
    viewport_cam_list = pegasus_setup.create_camera_trajectory(num_cameras=100, num_interpolation_steps=2)

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
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    pegasus_setup.init_video_streams(output='./output', fps=10)

    with (torch.no_grad()):
        gaussian_environment = GaussianModel(dataset.sh_degree)
        gaussian_environment.load_ply(pegasus_setup.environment.gaussian_point_cloud_path(iteration=load_iteration))

        # scene = Scene(dataset, gaussian_environment, load_iteration=load_iteration, shuffle=False)

        gaussians_object_list = pegasus_setup.load_object_gs(sh_degree=dataset.sh_degree, load_iteration=load_iteration)

        # Static
        if pegasus_setup.mode == 'static':
            gaussians_object_list = pegasus_setup.static_object_pose(gaussians_object_list=gaussians_object_list)
        # Dynamic
        elif pegasus_setup.mode == 'dynamic':
            gaussians_object_list = pegasus_setup.init_object_pose(gaussians_object_list=gaussians_object_list)
        else:
            raise ValueError('Mode -{}- not available'.format(pegasus_setup.mode))
        semantic_colors = generate_colors(len(gaussians_object_list))

        if network_gui.conn is None:
            network_gui.try_connect()

        bar = tqdm.tqdm(total=len(viewport_cam_list))
        for i in range(len(viewport_cam_list) + 1):
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
                # Set camera from camera trajectory
                viewpoint_cam = viewport_cam_list[i]

                rgb_image = None
                depth_image = None
                mask_silhouette = None
                individual_seg_masks = None
                seg_image = None
                object_center_image = None

                if 'rgb' in data_points:
                    # Render rgb
                    rgb_image, depth_image = render_rgb_and_depth(cam=viewpoint_cam,
                                                                  gs_scene=gaussian_scene,
                                                                  pipe_settings=pipe,
                                                                  bg=background,
                                                                  debug=False)

                if 'seg_sil' in data_points:
                    # Render silhouette mask
                    mask_silhouette = render_silhouette_mask(cam=viewpoint_cam,
                                                             gs_object_list=gaussians_object_list,
                                                             gs_env=gaussian_environment,
                                                             width=pegasus_setup.render_width,
                                                             height=pegasus_setup.render_height,
                                                             color_set=semantic_colors,
                                                             pipe_settings=pipe,
                                                             bg=background)
                if 'seg_vis' in data_points:
                    # Render visible mask
                    individual_seg_masks, seg_image = render_visib_mask(cam=viewpoint_cam,
                                                                        gs_environment=gaussian_environment,
                                                                        gs_object_list=gaussians_object_list,
                                                                        color_set=semantic_colors,
                                                                        width=pegasus_setup.render_width,
                                                                        height=pegasus_setup.render_height,
                                                                        pipe_settings=pipe,
                                                                        bg=background)
                #gaussian_scene.save_ply(path=r'C:\Users\meyerls\Documents\AIST\data\pegasus\environment\cobblestone_show\gs\point_cloud\merged\point_cloud.ply')
                if 'sem_seg' in data_points:
                    semantic_segmentation_mask = render_semanticsegmentation_mask(cam=viewpoint_cam,
                                                                                  gs_environment=gaussian_environment,
                                                                                  gs_object_list=gaussians_object_list,
                                                                                  color_set=semantic_colors,
                                                                                  width=pegasus_setup.render_width,
                                                                                  height=pegasus_setup.render_height,
                                                                                  pipe_settings=pipe,
                                                                                  bg=background,
                                                                                  debug=False)
                pegasus_dataset.add_scene_camera_json(frame_id=i)

                if save_bop:
                    pegasus_dataset.write_training_data(
                        rgb_image=(np.ascontiguousarray(rgb_image) * 255).astype('uint8'),
                        seg_image=individual_seg_masks,
                        mask_silhouette=mask_silhouette,
                        depth_image=(depth_image.numpy() * 1000).astype(np.uint16),
                        frame_id=i)  # meters in millimeter

                    pegasus_dataset.add_scene_gt_json(time_step=i,
                                                      gs_object_list=gaussians_object_list,
                                                      cam=viewpoint_cam,
                                                      rgb_image=(np.ascontiguousarray(rgb_image) * 255).astype('uint8'),
                                                      debug=False)

                #plt.imshow((np.ascontiguousarray(rgb_image) * 255).astype('uint8'))
                #plt.show()
                object_center_image = pegasus_setup.draw_object_center(
                    image=(np.ascontiguousarray(rgb_image) * 255).astype('uint8'),
                    gaussians_object_list=gaussians_object_list,
                    camera=viewpoint_cam,
                    semantic_colors=semantic_colors,
                    K=pegasus_dataset.K)
                if False:
                    imageio.imwrite('image_object_pose.png', object_center_image)
                    plt.imshow(object_center_image)
                    plt.show()

                # Save images to video stream
                pegasus_setup.write_image2video(rgb=(np.ascontiguousarray(rgb_image) * 255).astype('uint8'),
                                                depth=depth_image,
                                                seg=seg_image,
                                                center_image=object_center_image)
                bar.update(1)

            else:
                pegasus_setup.close_video_streams()
                pegasus_dataset.write_scene_camera_json()
                pegasus_dataset.write_scene_gt_json()
                print('Saved BOP data')

            if pegasus_setup.mode == 'dynamic':
                gaussians_object_list = pegasus_setup.update_object_pose(gaussians_object_list=gaussians_object_list,
                                                                         timestep=i + 1)


if __name__ == '__main__':
    # DATASET_PATH = 'C:\\Users\\meyerls\\Documents\\AIST\\data\\pegasus_dataset'
    DATASET_PATH = 'C:\\Users\\meyerls\\Documents\\AIST\\data\\pegasus'
    gaussian_splatting_manipulation(physics_file=r"C:\Users\meyerls\Documents\AIST\code\PEGASUS\dataset\pegasus_1\engine\000007_simulation_steps.json",
                                    dataset_path=DATASET_PATH,
                                    data_points=['rgb', 'depth', 'seg_vis', 'seg_sil', 'sem_seg'],
                                    # data_points=['rgb'],
                                    save_bop=True,
                                    scene_id=1,
                                    render_height=480,
                                    render_width=640,
                                    IP="127.0.0.1",
                                    PORT=6010,
                                    )
