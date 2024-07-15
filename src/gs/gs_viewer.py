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

sys.path.append("C:/Users/meyerls/Documents/AIST/code/pegasus/gaussian-splatting")

import torch
from gaussian_renderer import render, network_gui
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from src.gs.gaussian_model import GaussianModel

def gaussian_splatting_viewer(data_path: str,
                              model_path: str,
                              object,
                              load_iteration: int = 30_000,
                              IP: str = "127.0.0.1",
                              PORT: int = 6009,
                              QUIET: bool = False):
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    sys.argv.append('-m')
    sys.argv.append(model_path)

    args = get_combined_args(parser)

    print("Rendering " + model_path)

    network_gui.init(IP, PORT)

    # Initialize system state (RNG)
    safe_state(QUIET)
    pipe = pipeline.extract(args)
    # render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)

    dataset = model.extract(args)

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        gaussians.load_ply(object.gaussian_point_cloud_path(iteration=load_iteration))

        #scene = Scene(dataset, gaussians, load_iteration=load_iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    if network_gui.conn is None:
        network_gui.try_connect()
    while network_gui.conn is not None:
        try:
            net_image_bytes = None
            custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
            if custom_cam is not None:
                net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2,
                                                                                                           0).contiguous().cpu().numpy())
            network_gui.send(net_image_bytes, dataset.source_path)

        except Exception as e:
            network_gui.conn = None
            print("Termination error: ", e)


if __name__ == '__main__':
    from src.dataset_objects import *

    DATASET_PATH = 'C:\\Users\\meyerls\\Documents\\AIST\\data\\pegasus'
    object = CrackerBox(dataset_path=DATASET_PATH)
    object.mode = 'fused'

    reco_path = object.reconstruction_path
    gs_path = object.gs_model_path

    gaussian_splatting_viewer(data_path=reco_path, model_path=gs_path, object=object)
