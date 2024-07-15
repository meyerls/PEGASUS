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
import tqdm
import json
import pylab as plt
import os
import cv2
from pathlib import Path
import imageio
import torch
import numpy as np

sys.path.append("./submodules/colmap-wrapper")
sys.path.append("./submodules/gaussian-splatting-pegasus")

from gaussian_renderer import render
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from src.gs.gaussian_model import GaussianModel
import src.dataset.dataset_envs as env_assets
from src.utility.graphic_utils import *
from utils.graphics_utils import focal2fov, fov2focal
import src.dataset as object_assets
from scene.cameras import Camera
from scipy.spatial.transform import Rotation
from spatialmath import *

from colmap_wrapper.dataloader import (read_images_binary, read_cameras_binary)

def compute_rotation_between_matrices(A, B):
    """
    Computes the rotation matrix that rotates the coordinate system of A to that of B.

    :param A: A 4x4 transformation matrix representing the first coordinate system.
    :param B: A 4x4 transformation matrix representing the second coordinate system.
    :return: A 3x3 rotation matrix that rotates A to B.
    """
    # Extract the upper-left 3x3 rotation components of A and B
    A_rot = A[:3, :3]
    B_rot = B[:3, :3]

    # Compute the rotation from A to B
    R = B_rot @ np.linalg.inv(A_rot)

    return R

def rotate_camera_around_origin_z(radius=0.5, steps=10):
    # List to store the extrinsic matrices
    extrinsics = []

    # Calculate the angle increment for each step
    angle_increment = 2 * np.pi / steps

    # Create a basic rotation matrix around the y-axis
    def rotation_matrix_z(theta):
        return np.array([
            [np.cos(theta), -np.sin(theta), 0, 0],
            [np.sin(theta), np.cos(theta), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

    # Create a translation matrix to move the camera back to the given radius
    translation_matrix = np.array([
        [-1, 0, 0, 0],
        [0, 0, -1, 0],
        [0, -1, 0, radius],
        [0, 0, 0, 1]
    ])

    # Calculate extrinsic matrices for each step
    for step in range(steps):
        # Current angle for this step
        theta = step * angle_increment

        # Calculate rotation matrix for current angle
        rot_matrix = rotation_matrix_z(theta)

        # Combine rotation and translation to get the extrinsic matrix
        # Note: In real-world applications, you might need to adjust the order of multiplication
        # depending on the convention (column-major vs row-major) and whether you're using
        # right-handed or left-handed coordinate systems.
        extrinsic_matrix = np.matmul(rot_matrix, translation_matrix)

        # Add the extrinsic matrix to the list
        extrinsics.append(extrinsic_matrix)

    return extrinsics


def plot_3d(M, horizontal=[-1, 1], vertical=[-1, 1], length=1, frame=True):
    plt.figure(figsize=(15, 15), dpi=150)  # create a new figure
    SE3().plot(dims=[horizontal[0], horizontal[1], horizontal[0], horizontal[1], vertical[0], vertical[1]],
               color='black', length=length)
    Line3.PointDir([0, 0, 0], [0, 0, 1]).plot()
    for i, m in enumerate(M):
        if frame:
            SE3(m).plot(frame=str(i), length=length, color=['red', 'green', 'blue'], labels=[""] * 3)
        else:
            SE3(m).plot(length=length, color=['red', 'green', 'blue'], labels=[""] * 3)
    plt.show()


class PegasusSetup(object):
    def __init__(self, pybullet_trajectory_path, dataset_path, render_height: int, render_width: int):
        self.pybullet_trajectory_path = pybullet_trajectory_path
        self.pyhsics_data = self.load_json(file=pybullet_trajectory_path)

        self.dataset_path = dataset_path

        # Select env parameters
        environment = self.pyhsics_data['asset_infos']['environment']
        self.environment_name = list(environment.keys())[0]  # select dataset name
        self.environment_class_name = environment[self.environment_name]['class_name']
        self.environment = getattr(env_assets, self.environment_class_name)(dataset_path="/media/se86kimy/PortableSSD/data/pegasus")

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

    def create_camera_trajectory(self, radius=0.4, num_interpolation_steps=24):
        cams = []

        focal_length_x = np.asarray(self.camera_data[0]['fx'])
        focal_length_y = np.asarray(self.camera_data[0]['fx'])
        FovY = focal2fov(focal_length_y, self.cam_intr[1].height)
        FovX = focal2fov(focal_length_x, self.cam_intr[1].width)


        if True:
            camera_extrinsics = rotate_camera_around_origin_z(radius=radius, steps=num_interpolation_steps)

            for extrinsic in camera_extrinsics:
                R = extrinsic[:3, :3]
                t = np.array(extrinsic[:3, 3])

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

        #plot_3d([frame for frame in camera_extrinsics], length=0.1)


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
            for object_id in self.object_data[object_name]['id']:
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
                {str(obj_id + 1): {"diameter": self.calculate_diameter_of_object(obj=obj_mesh) * 1000,
                                   "min_x": axis_aligned_bbox.min_bound[0] * 1000,
                                   "min_y": axis_aligned_bbox.min_bound[1] * 1000,
                                   "min_z": axis_aligned_bbox.min_bound[2] * 1000,
                                   "size_x": axis_aligned_bbox.get_extent()[0] * 1000,
                                   "size_y": axis_aligned_bbox.get_extent()[1] * 1000,
                                   "size_z": axis_aligned_bbox.get_extent()[2] * 1000}})

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
            if debug:  # debug
                mesh = o3d.io.read_triangle_mesh(gs_object_list[object_id].meta_info.urdf_obj_path)
                # mesh = mesh.translate(mesh.get_center())
                coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                # o3d.visualization.draw_geometries([mesh, coord])

                points = np.ones((100, 4))
                points[:, :3] = np.asarray(mesh.sample_points_uniformly(points.shape[0]).points)
                # points[:, :3] = gaussians_object_list[1]._xyz[:points.shape[0]].cpu().numpy()

            # Model to World transformation
            T_m2w = np.eye(4)
            T_m2w[:3, :3] = gs_object_list[object_id].R_init.cpu().numpy()
            T_m2w[:3, -1] = gs_object_list[object_id].t_init.cpu().numpy()  # - mesh.get_center()

            T = T_w2c @ T_m2w

            if debug:  # debug
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

            self.scene_gt_json[str(time_step)].append({
                'cam_R_m2c': list(R_m2c.flatten()),
                'cam_t_m2c': list(t_m2c.flatten()),
                'obj_id': object_id,
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


def gaussian_splatting_viewer(data_path: str,
                              object,
                              dataset_path: str,
                              model_path: str,
                              render_height: int = 480*2,
                              render_width: int = 640*2,
                              load_iteration: int = 30_000,
                              IP: str = "127.0.0.1",
                              PORT: int = 6009,
                              QUIET: bool = False):
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    pegasus_setup = PegasusSetup(pybullet_trajectory_path='src/engine/simulation_steps.json',
                                 dataset_path=dataset_path,
                                 render_height=render_height,
                                 render_width=render_width)
    viewport_cam_list = pegasus_setup.create_camera_trajectory(radius=0.4, num_interpolation_steps=250)

    sys.argv.append('-m')
    sys.argv.append(model_path)

    args = get_combined_args(parser)

    print("Rendering " + model_path)

    #network_gui.init(IP, PORT)

    # Initialize system state (RNG)
    safe_state(QUIET)
    pipe = pipeline.extract(args)
    # render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)

    dataset = model.extract(args)

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        gaussians.load_ply(object.gaussian_point_cloud_path(iteration=load_iteration))

        # scene = Scene(dataset, gaussians, load_iteration=load_iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    rgb_name = os.path.join("./output_objects", '{}.mp4'.format(object.object_name))
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    image_size = (render_width, render_height)
    fps = 30

    rgb_video_writer = cv2.VideoWriter(rgb_name, fourcc, fps, image_size)

    bar = tqdm.tqdm(total=len(viewport_cam_list))
    for i in range(len(viewport_cam_list) + 1):
        if i < len(viewport_cam_list):
            # Set camera from camera trajectory
            viewpoint_cam = viewport_cam_list[i]
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        net_image, depth_image = render_pkg["render"], render_pkg["depth"]

        rgb_image = net_image.cpu().permute((1, 2, 0)).detach().numpy()
        rgb_image = (np.ascontiguousarray(rgb_image) * 255).astype('uint8')
        rgb_video_writer.write(cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
        bar.update(1)


if __name__ == '__main__':
    from src.dataset.ycb_objects import *
    from src.dataset.in_the_wild_dataset import *
    from src.dataset.cup_noodle_dataset import *

    DATASET_PATH = './workspace'

    object_list = []
    if False:
        # YCB
        object_list.append(Banana(dataset_path=DATASET_PATH))
        object_list.append(ChocoJello(dataset_path=DATASET_PATH))
        object_list.append(CrackerBox(dataset_path=DATASET_PATH))
        object_list.append(DominoSugar(dataset_path=DATASET_PATH))
        object_list.append(MaxwellCoffee(dataset_path=DATASET_PATH))
        object_list.append(Pitcher(dataset_path=DATASET_PATH))
        object_list.append(RedBowl(dataset_path=DATASET_PATH))
        object_list.append(RedCup(dataset_path=DATASET_PATH))
        object_list.append(SoftScrub(dataset_path=DATASET_PATH))
        object_list.append(TomatoSoup(dataset_path=DATASET_PATH))
        object_list.append(WoodenBlock(dataset_path=DATASET_PATH))
        object_list.append(YellowMustard(dataset_path=DATASET_PATH))
        object_list.append(Spam(dataset_path=DATASET_PATH))
        object_list.append(StrawberryJello(dataset_path=DATASET_PATH))
        object_list.append(Tuna(dataset_path=DATASET_PATH))
        object_list.append(Pen(dataset_path=DATASET_PATH))
        object_list.append(FoamBrick(dataset_path=DATASET_PATH))
        object_list.append(Scissors(dataset_path=DATASET_PATH))

        # CupNoodles
        object_list.append(CupNoodle01(dataset_path=DATASET_PATH))
        object_list.append(CupNoodle02(dataset_path=DATASET_PATH))
        object_list.append(CupNoodle03(dataset_path=DATASET_PATH))
        object_list.append(CupNoodle04(dataset_path=DATASET_PATH))
        object_list.append(CupNoodle05(dataset_path=DATASET_PATH))
        object_list.append(CupNoodle06(dataset_path=DATASET_PATH))
        object_list.append(CupNoodle07(dataset_path=DATASET_PATH))
        object_list.append(CupNoodle08(dataset_path=DATASET_PATH))
        object_list.append(CupNoodle09(dataset_path=DATASET_PATH))
        object_list.append(CupNoodle10(dataset_path=DATASET_PATH))
        object_list.append(CupNoodle11(dataset_path=DATASET_PATH))
        object_list.append(CupNoodle12(dataset_path=DATASET_PATH))
        object_list.append(CupNoodle13(dataset_path=DATASET_PATH))
        object_list.append(CupNoodle14(dataset_path=DATASET_PATH))
        object_list.append(CupNoodle15(dataset_path=DATASET_PATH))
        object_list.append(CupNoodle16(dataset_path=DATASET_PATH))
        object_list.append(CupNoodle17(dataset_path=DATASET_PATH))
        object_list.append(CupNoodle18(dataset_path=DATASET_PATH))
        object_list.append(CupNoodle19(dataset_path=DATASET_PATH))
        object_list.append(CupNoodle20(dataset_path=DATASET_PATH))
        object_list.append(CupNoodle21(dataset_path=DATASET_PATH))
        object_list.append(CupNoodle22(dataset_path=DATASET_PATH))
    #object_list.append(CupNoodle23(dataset_path=DATASET_PATH))
    #object_list.append(CupNoodle24(dataset_path=DATASET_PATH))
    #object_list.append(CupNoodle20(dataset_path=DATASET_PATH))
    #object_list.append(SmallClamp(dataset_path=DATASET_PATH))
    #object_list.append(LargeClamp(dataset_path=DATASET_PATH))
    object_list.append(Bouillon(dataset_path=DATASET_PATH))

    for object in object_list:
        object.mode = 'fused'

        reco_path = object.reconstruction_path
        gs_path = object.gs_model_path

        gaussian_splatting_viewer(data_path=reco_path, model_path=gs_path, dataset_path=DATASET_PATH, object=object)
