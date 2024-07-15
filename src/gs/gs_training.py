import sys
import torch
from argparse import ArgumentParser

sys.path.append("./submodules/gaussian-splatting-pegasus")

from train import training
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.general_utils import safe_state
from gaussian_renderer import network_gui


def train_gaussian_splatting_wrapper(data_path: str,
                                     model_path: str,
                                     IP: str = "127.0.0.1",
                                     gui: bool = True,
                                     PORT: int = 6009,
                                     DEBUG_FROM: int = -1,
                                     DETECT_ANOMALY: bool = False,
                                     TEST_ITERATION: list = [7_000, 30_000],
                                     SAVE_ITERATION: list = [7_000, 30_000],
                                     QUIET: bool = False,
                                     CHECKPOINT_ITERATION: list = [],
                                     START_CHECKPOINT=None,
                                     ):
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")

    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    args = parser.parse_args(sys.argv[1:])
    args.source_path = data_path
    args.model_path = model_path

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(QUIET)

    # Start GUI server, configure and run training
    if gui:
        network_gui.init(IP, PORT)
    torch.autograd.set_detect_anomaly(DETECT_ANOMALY)
    training(lp.extract(args), op.extract(args), pp.extract(args), TEST_ITERATION, SAVE_ITERATION,
             CHECKPOINT_ITERATION, START_CHECKPOINT, DEBUG_FROM)

    # All done
    print("\nTraining complete.")


if __name__ == '__main__':
    from src.dataset.dataset_envs import Garden

    DATASET_PATH = '/media/se86kimy/PortableSSD/data/pegasus'
    object = Garden(dataset_path=DATASET_PATH)

    reco_path = object.reconstruction_path
    gs_path = object.gs_model_path

    train_gaussian_splatting_wrapper(data_path=reco_path, model_path=gs_path)
