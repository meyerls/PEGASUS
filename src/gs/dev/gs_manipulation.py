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
from gaussian_renderer import render, network_gui
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
import json
from scipy.spatial.transform import Rotation
import copy
import src.dataset.dataset_envs as env_assets
import src.dataset_objects as object_assets


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

    network_gui.init(IP, PORT)

    # Initialize system state (RNG)
    safe_state(QUIET)
    pipe = pipeline.extract(args)
    # render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)

    dataset = model.extract(args)

    with torch.no_grad():
        gaussian_environment = GaussianModel(dataset.sh_degree)
        gaussian_environment.load_ply(env.gaussian_point_cloud_path(iteration=load_iteration))

        #scene = Scene(dataset, gaussian_environment, load_iteration=load_iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        gaussians_object_list = {}
        for object_name in data_loaded['asset_infos']['object'].keys():
            for id in data_loaded['asset_infos']['object'][object_name]['id']:
                obj_class_name = data_loaded['asset_infos']['object'][object_name]['class_name']
                obj = getattr(object_assets, obj_class_name)(dataset_path=dataset_path)

                obj.mode = 'fused'
                gs_object = GaussianModel(dataset.sh_degree)
                gs_object.load_ply(obj.gaussian_point_cloud_path(iteration=load_iteration))

                t_init = torch.from_numpy(-np.asarray(data_loaded['trajectory'][str(id)][str(0)]['t'])).type(torch.float32).to(
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
                    q_t_0 = Rotation.from_quat(np.asarray(data_loaded['trajectory'][str(gs_object_id)][str(i - 1)]['q']))
                    q_delta = q_t_1 * q_t_0.inv()
                    R_delta = torch.from_numpy(q_delta.as_matrix()).type(torch.float32).to(device="cuda")
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


if __name__ == '__main__':
    #from src.dataset_objects import WoodenBlock, CupNoodle01
    #from src.dataset_envs import PlainTableSetup_01, PlainTableSetup_02, Garden

    DATASET_PATH = 'C:\\Users\\meyerls\\Documents\\AIST\\data\\pegasus_dataset'

    gaussian_splatting_manipulation(physics_file='../../engine/simulation_steps.json', dataset_path=DATASET_PATH)
