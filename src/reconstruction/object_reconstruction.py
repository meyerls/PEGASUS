from src.dataset.data_sfm_reconstruction import COLMAPReconstruction
from src.dataset.data_alignment import ReconstructionAlignment
from src.gs.gs_training import train_gaussian_splatting_wrapper
from src.dataset.data_urdf import URDFGenerator

import sys
from pathlib import Path
import os
import numpy as np
import shutil

# sys.path.append("C:/Users/meyerls/Documents/AIST/code/gaussian-splatting/aruco-estimator")
sys.path.append("./submodules/colmap-wrapper")

# from aruco_estimator.aruco_scale_factor import ArucoScaleFactor
# from aruco_estimator.visualization import ArucoVisualization

from colmap_wrapper.dataloader import COLMAPLoader
from colmap_wrapper.visualization import ColmapVisualization

from src.dataset.ycb_objects import *
from src.dataset.in_the_wild_dataset import *
from src.dataset.cup_noodle_dataset import *
from src.dataset.dataset_envs import *

import distutils.spawn

COLMAP_EXE = distutils.spawn.find_executable("colmap")
if COLMAP_EXE is None:
    COLMAP_EXE = None  # Set path to colmap here!
    raise FileNotFoundError("colmap not found. Please set path manually!")


class ObjectReconstruction:
    def __init__(self, reco_object, DEBUG=False):
        self.reco_object = reco_object
        self.DEBUG = DEBUG

    def preprocess(self):
        # In the wild, spherical or hemispherical pre-processing of data
        self.reco_object.prepare_dataset()

    def run(self):

        if self.reco_object.DATASET_TYPE is ("cup_noodles" or "ycb"):
            # These two dataset use the ortery system and have thus a calibration board:
            database_name = 'database_0.db'
        else:
            database_name = 'database.db'

        if self.reco_object.CALIBRATION_OBJECT is not None:
            # Check if the object has a calibration pattern (this is the case for the ortery reconstructions)
            reference_reconstruction = self.reco_object.CALIBRATION_OBJECT(
                dataset_path=self.reco_object.dataset_path).reconstruction_path
        else:
            reference_reconstruction = None

        if self.reco_object.RECORDING_TYPE == "spherical":
            # If reconstruction type is spherical we have to compute the upper reco and integrate the bottom
            # reconstruction into the upper one. Therefore, we set the reconstruction type "up". This is needed as
            # spherical data have a different folder structure.
            self.reco_object.mode = 'up'
            output_path = Path(self.reco_object.fused_path)
        else:
            # For hemispherical and environment reconstruction
            output_path = Path(self.reco_object.reconstruction_path)

        if self.reco_object.DATASET_TYPE is ("cup_noodles" or "ycb" or "wild"):
            image_list_path = Path(reconstruction_object.output_path) / 'image_list.txt'
        else:
            image_list_path = None

        # Compute poses for upper spherical or hemispherical data
        gs_reco_up = COLMAPReconstruction(image_path=Path(self.reco_object.image_masked_path),  # masked_png_path
                                          output_path=output_path,
                                          camera_model=self.reco_object.camera_model,
                                          database_name=database_name,
                                          resize=self.reco_object.resize,
                                          single_camera=True,  # Do not change this!!!
                                          gpu=True,
                                          colmap_exe=COLMAP_EXE,
                                          )
        gs_reco_up.run(reference_reconstruction=reference_reconstruction,
                       image_list_path=image_list_path)

        # Just for visual check how the reconstruction looks like. Visually checking for outliers to relax or tighten
        # the colmap reconstruction parameters
        if self.DEBUG:
            if self.reco_object.RECORDING_TYPE == "spherical":
                reco_path = Path(self.reco_object.fused_path)
            elif self.reco_object.RECORDING_TYPE == "hemispherical":
                reco_path = self.reco_object.dataset_path / self.reco_object.object_name
            elif self.reco_object.DATASET_TYPE == "environment":
                reco_path = Path(reconstruction_object.reconstruction_path)
            else:
                raise ValueError("")

            project_reference = COLMAPLoader(project_path=reco_path,
                                             img_orig=reco_path / 'images',
                                             sparse_folder_path=(reco_path / 'sparse/0').__str__(),
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
                                     image_name=os.path.join(reco_path,
                                                             "{}.png".format(self.reco_object.object_name)))

        if self.reco_object.DATASET_TYPE is ("wild" or "environment"):
            if isinstance(reconstruction_object.SCALE, bool) and self.reco_object.SCALE == True:
                gs_reco_up.scale_scene(aruco_scale=self.reco_object.ARUCO_SIZE,
                                       img_orig=self.reco_object.orig_path,
                                       visualize=self.DEBUG)
            elif isinstance(self.reco_object.SCALE, float):
                gs_reco_up.scale_scene_by_const(scale=self.reco_object.SCALE)
            else:
                print('No scaling is applied or already scaled with calibration board!')

            if self.reco_object.RECORDING_TYPE == "spherical":
                reco_path = Path(self.reco_object.fused_path)
            elif self.reco_object.DATASET_TYPE == "environment":
                reco_path = Path(self.reco_object.reconstruction_path)

            # Align environment to xy plane
            reco_align = ReconstructionAlignment(project_path=reco_path)
            reco_align.align2plane(plane_size=2.,
                                   plane_normal=np.asarray(reconstruction_object.PLANE_NORMAL),
                                   debug=False)
            # reco_align.visualize(add_object=[reco_align.plane_mesh], coord_system=True)
            reco_align.save()

        # Registrate bottom hemisphere into top one
        if self.reco_object.RECORDING_TYPE == "spherical":
            # Copy rest of the images to fused folder
            self.reco_object.mode = 'down'
            shutil.copytree(self.reco_object.image_masked_path,
                            (Path(self.reco_object.fused_path) / 'images').__str__(),
                            dirs_exist_ok=True)
            #
            #
            gs_reco_up.registrate_images_into_existing_model(
                database_path=(Path(self.reco_object.fused_path) / 'distorted/database.db').__str__(),
                working_dir_images=(Path(self.reco_object.fused_path) / 'images').__str__(),
                image_list_path=(Path(self.reco_object.output_path) / 'image_list.txt').__str__().replace('\\up\\',
                                                                                                          '\\down\\'),
                sparese_model_path=(Path(self.reco_object.fused_path) / 'sparse/0').__str__(),
                output_path=(Path(self.reco_object.fused_path) / 'sparse/0').__str__(),
                image_registration_mapper_settings=self.reco_object.MATCHING)

            if self.DEBUG:
                reco_path = Path(self.reco_object.fused_path)
                project_reference = COLMAPLoader(project_path=reco_path,
                                                 img_orig=reco_path / 'images',
                                                 sparse_folder_path=(reco_path / 'sparse/0').__str__(),
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
                                         image_name=os.path.join(reco_path,
                                                                 "{}.png".format(self.reco_object.object_name)))
        if self.reco_object.RECORDING_TYPE == "spherical":
            reco_path = Path(self.reco_object.fused_path)
        elif self.reco_object.RECORDING_TYPE == "hemispherical":
            reco_path = self.reco_object.dataset_path / self.reco_object.object_name
        elif self.reco_object.DATASET_TYPE == "environment":
            reco_path = Path(self.reco_object.reconstruction_path)
        else:
            raise ValueError("")

        # Train gaussian splatting on object
        train_gaussian_splatting_wrapper(data_path=reco_path,
                                         model_path=(Path(reco_path) / 'gs').__str__(),
                                         TEST_ITERATION=[7_000, 30_000],
                                         SAVE_ITERATION=[7_000, 30_000],
                                         gui=False
                                         )

        if self.reco_object.RECORDING_TYPE == "spherical":
            self.reco_object.mode = 'fused'

        if self.reco_object.DATASET_TYPE is ("cup_noodles" or "ycb" or "wild"):
            urdf_template = './src/dataset/urdf_object_template.urdf'
        else:
            urdf_template = './src/dataset/urdf_environment_template.urdf'

        gen = URDFGenerator(object_path=self.reco_object.gs_o3d_point_cloud_path(iteration=30_000),
                            urdf_template=urdf_template,
                            object_type=self.reco_object.TYPE,
                            meta_info=self.reco_object,
                            ycb_path=self.reco_object.REFERENCE_DATASET_PATH
                            )
        gen.generate(obj_path=self.reco_object.urdf_obj_path,
                     urdf_path=self.reco_object.urdf_file_path,
                     alpha=self.reco_object.ALPHA)

        self.reco_object.gs_cleaning(t=gen.center_translation, R=gen.center_rotation)


if __name__ == '__main__':
    CUP_NOODLE_DATASET_BASE_PATH = '/media/se86kimy/PortableSSD/data/pegasus'
    YCB_DATASET_BASE_PATH = '/media/se86kimy/PortableSSD/data/pegasus'
    ENV_BASE_PATH = '/media/se86kimy/PortableSSD/data/pegasus'
    IN_THE_WILD_DATASET_BASE_PATH = './workspace'

    DEBUG = False
    reco_list = []

    #reco_list.append(CupNoodleTEST(dataset_path=CUP_NOODLE_DATASET_BASE_PATH))
    #reco_list.append(Scissors(dataset_path=YCB_DATASET_BASE_PATH))
    reco_list.append(Bouillon(dataset_path=IN_THE_WILD_DATASET_BASE_PATH))
    #reco_list.append(MannholeCover(dataset_path=ENV_BASE_PATH))

    for reconstruction_object in reco_list:
        reco = ObjectReconstruction(reconstruction_object)
        reco.preprocess()
        reco.run()
