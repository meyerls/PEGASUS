from src.dataset.data_alignment import ReconstructionAlignment
from src.dataset.data_ortery_preperation import OrteryImageProcessor
from src.dataset.data_sfm_reconstruction import COLMAPReconstruction
from src.gs.gs_training import train_gaussian_splatting_wrapper
from src.dataset.data_urdf import URDFGenerator
import os
import sys
import open3d as o3d
from pathlib import Path
import copy

# sys.path.append("C:/Users/meyerls/Documents/AIST/code/gaussian-splatting/aruco-estimator")
sys.path.append("./submodules/colmap-wrapper")

from colmap_wrapper.dataloader import COLMAPLoader
from colmap_wrapper.visualization import ColmapVisualization

MAGICK_EXE = "C:\\Program Files\\ImageMagick-7.1.1-Q16-HDRI\\magick.exe"
COLMAP_EXE = "/usr/local/bin/colmap"
DATASET_BASE_PATH = '/home/se86kimy/Documents/data/PEGASET'
YCB_MODEL_PATH = r"/home/se86kimy/Documents/data/ycbv"

DEBUG = True

from src.dataset.ycb_objects import *
from src.dataset.cup_noodle_dataset import *

# reconstruction_object = WoodenBlock(dataset_path=DATASET_BASE_PATH)
# reconstruction_object = Drill(dataset_path=DATASET_BASE_PATH)
# reconstruction_object = SmallClamp(dataset_path=DATASET_BASE_PATH)
# reconstruction_object = DominoSugar(dataset_path=DATASET_BASE_PATH)
# reconstruction_object = LargeClamp(dataset_path=DATASET_BASE_PATH)
# reconstruction_object.mode = 'up' # Set by default

# Preprocessing images
ortery = OrteryImageProcessor(orig_folder=Path(reconstruction_object.orig_path),
                              masked_folder=Path(reconstruction_object.masked_png_path),
                              mask_folder=Path(reconstruction_object.mask_path),
                              output_path=Path(reconstruction_object.output_path),
                              debug=False)
ortery.process()

# Compute poses for environment
gs_reco = COLMAPReconstruction(image_path=Path(reconstruction_object.image_masked_path),  # masked_png_path
                               output_path=Path(reconstruction_object.reconstruction_path),
                               camera_model=reconstruction_object.camera_model,
                               resize=reconstruction_object.resize,
                               single_camera=True,  # Do not change this!!!
                               gpu=True,
                               colmap_exe=COLMAP_EXE,
                               magick_exe=MAGICK_EXE
                               )
if reconstruction_object.CALIBRATION_OBJECT:
    calibration_object_reconstruction_path = reconstruction_object.CALIBRATION_OBJECT(
        dataset_path=DATASET_BASE_PATH).reconstruction_path
else:
    calibration_object_reconstruction_path = None

gs_reco.run(reference_reconstruction=calibration_object_reconstruction_path,
            image_list_path=Path(reconstruction_object.output_path) / 'image_list.txt')

if DEBUG:
    reco_path = reconstruction_object.dataset_path / reconstruction_object.object_name
    project_reference = COLMAPLoader(project_path=reco_path,
                                     img_orig=reconstruction_object.image_masked_path,
                                     sparse_folder_path=(
                                             Path(reconstruction_object.reconstruction_path) / 'sparse/0').__str__(),
                                     load_sparse_only=True,
                                     load_depth=False)
    reco_project = project_reference.project_list[0]

    project_vs = ColmapVisualization(reco_project)
    perspective = {
        "front": [-0.09478730993217184, 0.96068117989395907, 0.26097324857803539],
        "lookat": [-0.071497577749656052, 0.0045387451776697646, 0.10277239081935101],
        "up": [0.055608993741667882, -0.25663456673739032, 0.96490742507794558],
        "zoom": 0.69999999999999996
    }
    project_vs.visualization(frustum_scale=0.1,
                             image_type='image',
                             show_dense=False,
                             point_size=2,
                             perspective=perspective,
                             show=True,
                             image_name=os.path.join(reco_path, "{}.png".format(reconstruction_object.object_name)))

# Train gaussian splatting on object
train_gaussian_splatting_wrapper(data_path=reconstruction_object.reconstruction_path,
                                 model_path=reconstruction_object.gs_model_path,
                                 TEST_ITERATION=[7_000, 30_000],
                                 SAVE_ITERATION=[7_000, 30_000],
                                 )

gen = URDFGenerator(object_path=reconstruction_object.gs_o3d_point_cloud_path(iteration=30_000),
                    urdf_template='./src/dataset/urdf_object_template.urdf',
                    meta_info=reconstruction_object,
                    ycb_path=YCB_MODEL_PATH,
                    object_type=reconstruction_object.TYPE)

gen.generate(obj_path=reconstruction_object.urdf_obj_path,
             urdf_path=reconstruction_object.urdf_file_path,
             alpha=reconstruction_object.ALPHA)

reconstruction_object.gs_cleaning(t=gen.center_translation, R=gen.center_rotation)
