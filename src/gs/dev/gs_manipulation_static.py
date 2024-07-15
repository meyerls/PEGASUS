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

from gaussian_renderer import render, network_gui
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import json
from scipy.spatial.transform import Rotation
import copy
import src.dataset.dataset_envs as env_assets
import src.dataset_objects as object_assets
from src.utility.pose_interpolation import interpolate_pose
from scene.cameras import Camera
# from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary
from utils.graphics_utils import focal2fov
from pathlib import Path
import cv2
from src.utility.graphic_utils import *

sys.path.append("C:/Users/meyerls/Documents/AIST/code/gaussian-splatting/colmap-wrapper")

from colmap_wrapper.dataloader import (read_images_binary, read_cameras_binary)


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

    def static_object_pose(self, gaussians_object_list: dict):

        for object_id in gaussians_object_list.keys():
            last_time_step = list(self.object_trajectory[str(1)].keys())[-1]

            t_init = -torch.asarray((self.object_trajectory[str(object_id)][str(last_time_step)]['t']),
                                    dtype=torch.float32, device='cuda')

            q = Rotation.from_quat(np.asarray(self.object_trajectory[str(object_id)][str(last_time_step)]['q']))
            euler = q.as_euler('xyz')
            # WTF?! rotations works but i cant explain it
            euler[1] *= -1
            euler[2] *= -1
            # euler[0] *= -1
            q_delta = Rotation.from_euler('xyz', euler)

            R_delta = torch.from_numpy(q_delta.as_matrix()).type(torch.float32).to(device="cuda")
            # R_delta = R_delta.T

            gaussians_object_list[object_id].center_position = t_init
            gaussians_object_list[object_id].rotation_matrix = q.as_matrix()

            gaussians_object_list[object_id].apply_translation_on_xyz(t=t_init)
            gaussians_object_list[object_id].apply_rotation_on_xyz(R=R_delta)
            gaussians_object_list[object_id].apply_rotation_on_splats(R=R_delta)
            gaussians_object_list[object_id].apply_rotation_on_sh(R=R_delta)

        return gaussians_object_list


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

    video_name = 'video_scene_origin.mp4'
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'MP4V'), 20,
                            (pegasus_setup.render_width, pegasus_setup.render_height))

    with torch.no_grad():
        gaussian_environment = GaussianModel(dataset.sh_degree)
        gaussian_environment.load_ply(pegasus_setup.environment.gaussian_point_cloud_path(iteration=load_iteration))

        # scene = Scene(dataset, gaussian_environment, load_iteration=load_iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        gaussians_object_list = pegasus_setup.load_object_gs(sh_degree=dataset.sh_degree, load_iteration=load_iteration)
        gaussians_object_list = pegasus_setup.static_object_pose(gaussians_object_list=gaussians_object_list)

        if False:
            gaussians_object_list = {}
            for object_name in data_loaded['asset_infos']['object'].keys():
                for id in data_loaded['asset_infos']['object'][object_name]['id']:
                    obj_class_name = data_loaded['asset_infos']['object'][object_name]['class_name']
                    obj = getattr(object_assets, obj_class_name)(dataset_path=dataset_path)

                    obj.mode = 'fused'
                    gs_object = GaussianModel(dataset.sh_degree)
                    gs_object.load_ply(obj.gaussian_point_cloud_path(iteration=load_iteration))
                    ##############

                    last_time_step = list(data_loaded['trajectory'][str(1)].keys())[-1]

                    t_init = torch.from_numpy(
                        -np.asarray(data_loaded['trajectory'][str(id)][str(last_time_step)]['t'])).type(
                        torch.float32).to(
                        device="cuda")

                    q = Rotation.from_quat(np.asarray(data_loaded['trajectory'][str(id)][str(last_time_step)]['q']))
                    euler = q.as_euler('xyz')
                    # WTF?! rotations works but i cant explain it
                    euler[1] *= -1
                    euler[2] *= -1
                    q_delta = Rotation.from_euler('xyz', euler)

                    R_delta = torch.from_numpy(q_delta.as_matrix()).type(torch.float32).to(device="cuda")
                    # R_delta = R_delta.T

                    gs_object.center_position = t_init
                    gs_object.rotation_matrix = q.as_matrix()

                    gs_object.apply_translation_on_xyz(t=t_init)
                    gs_object.apply_rotation_on_xyz(R=R_delta)
                    gs_object.apply_rotation_on_splats(R=R_delta)
                    gs_object.apply_rotation_on_sh(R=R_delta)

                    gaussians_object_list.update({id: gs_object})

        if network_gui.conn is None:
            network_gui.try_connect()

        i = 1
        # gs_base = copy.deepcopy(gaussian_environment)
        # gaussian_environment._xyz = torch.asarray([0, 0, 0], device='cuda', dtype=torch.float32)
        gaussian_scene = copy.deepcopy(gaussian_environment)

        for gs_object_id in gaussians_object_list.keys():
            gaussian_scene.merge_gaussians(gaussian=gaussians_object_list[gs_object_id])

        while network_gui.conn is not None:

            if False:
                if i < data_loaded['trajectory'][str(0)].__len__():
                    gaussian_scene = copy.deepcopy(gs_base)

                    for gs_object_id in gaussians_object_list.keys():
                        t_delta = torch.from_numpy(
                            np.asarray(data_loaded['trajectory'][str(gs_object_id)][str(i - 1)]['t']) - np.asarray(
                                data_loaded['trajectory'][str(gs_object_id)][str(i)]['t'])).type(
                            torch.float32).to(device="cuda")
                        q_t_1 = Rotation.from_quat(
                            np.asarray(data_loaded['trajectory'][str(gs_object_id)][str(i)]['q']))
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

            if i < len(viewport_cam_list):
                viewpoint_cam = viewport_cam_list[i]

            # custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
            render_pkg = render(viewpoint_cam, gaussian_scene, pipe, background)

            net_image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg[
                "viewspace_points"], \
                render_pkg["visibility_filter"], render_pkg["radii"]

            if i > 0 and i < len(viewport_cam_list):
                image = (np.ascontiguousarray(net_image.cpu().permute((1, 2, 0))) * 255).astype('uint8')
                if False:
                    for object_name in pegasus_setup.object_data.keys():
                        for id in pegasus_setup.object_data[object_name]['id']:
                            point = torch.asarray([[0, 0, 0, 1]], dtype=torch.float32, device='cuda')
                            object_offset = torch.asarray(pegasus_setup.object_data[object_name]['center_of_mass'],
                                                          dtype=torch.float32,
                                                          device='cuda')
                            point[0, :3] = gaussians_object_list[id].center_position - object_offset

                            camera_coord_hom = viewpoint_cam.world_view_transform.T @ point.T
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
                            a = viewpoint_cam.projection_matrix.T @ camera_coord_hom
                            b = (a.T[0] / a.T[0, -1])[:3]
                            b = b / b[-1]
                            # ndc = (camera_coord / camera_coord[-1])
                            screen_coord = ndc_to_screen(b[0], -b[1], width, height)

                            color = (255, 0, 0)  # BGR color
                            radius = 10  # Point radius (size)
                            # Draw a point on the image
                            image = cv2.circle(image, (int(screen_coord[0]), int(screen_coord[1])), radius, color,
                                               -1)  # -1 indicates a filled circle
                    # plt.imshow(image)
                    # plt.show()

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                video.write(image)
            elif i > len(viewport_cam_list):
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
