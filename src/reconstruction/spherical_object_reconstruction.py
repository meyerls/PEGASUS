from src.dataset.data_alignment import ReconstructionAlignment
from src.dataset.data_ortery_preperation import OrteryImageProcessor
from src.dataset.data_sfm_reconstruction import COLMAPReconstruction
from src.gs.gs_training import train_gaussian_splatting_wrapper
from src.dataset.data_urdf import URDFGenerator
from src.utility.colmap2nerf import convert_colmap2nerf

import sys
from pathlib import Path
import platform

import shutil
import os

# sys.path.append("C:/Users/meyerls/Documents/AIST/code/gaussian-splatting/aruco-estimator")
sys.path.append("./submodules/colmap-wrapper")

# from aruco_estimator.aruco_scale_factor import ArucoScaleFactor
# from aruco_estimator.visualization import ArucoVisualization

from colmap_wrapper.dataloader import COLMAPLoader
from colmap_wrapper.visualization import ColmapVisualization

MAGICK_EXE = "C:\\Program Files\\ImageMagick-7.1.1-Q16-HDRI\\magick.exe"
COLMAP_EXE = "/usr/local/bin/colmap"
# DATASET_BASE_PATH = 'C:\\Users\\meyerls\\Documents\\AIST\\data\\pegasus_dataset'
DATASET_BASE_PATH = '/home/se86kimy/Documents/data/PEGASET'
YCB_MODEL_PATH = r"/home/se86kimy/Documents/data/ycbv"

# from src.dataset_objects_old import *
from src.dataset.ycb_objects import *
from src.dataset.cup_noodle_dataset import *

DEBUG = True
reco_list = []

# reco_list.append(WoodenBlock(dataset_path=DATASET_BASE_PATH))
# reco_list.append(YellowMustard(dataset_path=DATASET_BASE_PATH))
# reco_list.append(RedBowl(dataset_path=DATASET_BASE_PATH))
# reco_list.append(DominoSugar(dataset_path=DATASET_BASE_PATH))
# reco_list.append(CrackerBox(dataset_path=DATASET_BASE_PATH))
# reco_list.append(ChocoJello(dataset_path=DATASET_BASE_PATH))
# reco_list.append(Banana(dataset_path=DATASET_BASE_PATH))
# reco_list.append(MaxwellCoffee(dataset_path=DATASET_BASE_PATH))
# reco_list.append(RedCup(dataset_path=DATASET_BASE_PATH))
# reco_list.append(Pitcher(dataset_path=DATASET_BASE_PATH))
# reco_list.append(SoftScrub(dataset_path=DATASET_BASE_PATH))
# reco_list.append(TomatoSoup(dataset_path=DATASET_BASE_PATH))
# reco_list.append(Spam(dataset_path=DATASET_BASE_PATH))
# reco_list.append(StrawberryJello(dataset_path=DATASET_BASE_PATH))
# reco_list.append(Tuna(dataset_path=DATASET_BASE_PATH))
# reco_list.append(Drill(dataset_path=DATASET_BASE_PATH))
# reco_list.append(FoamBrick(dataset_path=DATASET_BASE_PATH))
reco_list.append(Scissors(dataset_path=DATASET_BASE_PATH))
# reco_list.append(LargeClamp(dataset_path=DATASET_BASE_PATH))
# reco_list.append(Pen(dataset_path=DATASET_BASE_PATH))
# reco_list.append(Scissors(dataset_path=DATASET_BASE_PATH))
# reco_list.append(CupNoodle01(dataset_path=DATASET_BASE_PATH))
# reco_list.append(CupNoodle02(dataset_path=DATASET_BASE_PATH))
# reco_list.append(CupNoodle03(dataset_path=DATASET_BASE_PATH))
# reco_list.append(CupNoodle04(dataset_path=DATASET_BASE_PATH))
# reco_list.append(CupNoodle05(dataset_path=DATASET_BASE_PATH))
# reco_list.append(CupNoodle06(dataset_path=DATASET_BASE_PATH))
# reco_list.append(CupNoodle07(dataset_path=DATASET_BASE_PATH))
# reco_list.append(CupNoodle08(dataset_path=DATASET_BASE_PATH))
# reco_list.append(CupNoodle09(dataset_path=DATASET_BASE_PATH))
# reco_list.append(CupNoodle10(dataset_path=DATASET_BASE_PATH))
# reco_list.append(CupNoodle11(dataset_path=DATASET_BASE_PATH))
# reco_list.append(CupNoodle12(dataset_path=DATASET_BASE_PATH))
# reco_list.append(CupNoodle13(dataset_path=DATASET_BASE_PATH))
# reco_list.append(CupNoodle14(dataset_path=DATASET_BASE_PATH))
# reco_list.append(CupNoodle15(dataset_path=DATASET_BASE_PATH))
# reco_list.append(CupNoodle16(dataset_path=DATASET_BASE_PATH))
# reco_list.append(CupNoodle18(dataset_path=DATASET_BASE_PATH))
# reco_list.append(CupNoodle19(dataset_path=DATASET_BASE_PATH))
# reco_list.append(CupNoodle20(dataset_path=DATASET_BASE_PATH))
# reco_list.append(CupNoodle21(dataset_path=DATASET_BASE_PATH))
# reco_list.append(CupNoodle22(dataset_path=DATASET_BASE_PATH))
# reco_list.append(CupNoodle23(dataset_path=DATASET_BASE_PATH))
# reco_list.append(CupNoodle24(dataset_path=DATASET_BASE_PATH))
# reco_list.append(CupNoodle25(dataset_path=DATASET_BASE_PATH))
# reco_list.append(CupNoodle26(dataset_path=DATASET_BASE_PATH))
# reco_list.append(CupNoodle27(dataset_path=DATASET_BASE_PATH))
# reco_list.append(CupNoodle28(dataset_path=DATASET_BASE_PATH))
# reco_list.append(CupNoodle29(dataset_path=DATASET_BASE_PATH))
# reco_list.append(CupNoodle30(dataset_path=DATASET_BASE_PATH))


for reconstruction_object in reco_list:

    if True:
        reconstruction_object.RELEASE_MODE = False
        ####### Pre Processing Reconstruction
        reconstruction_object.mode = 'up'  # Set by default
        # Preprocessing images
        ortery = OrteryImageProcessor(orig_folder=Path(reconstruction_object.orig_path),
                                      masked_folder=Path(reconstruction_object.masked_png_path),
                                      mask_folder=Path(reconstruction_object.mask_path),
                                      output_path=Path(reconstruction_object.output_path),
                                      downscale_factor=2,
                                      debug=False)
        ortery.process(image_idx_start=1)

        reconstruction_object.mode = 'down'  # Set by default
        # Preprocessing images
        ortery = OrteryImageProcessor(orig_folder=Path(reconstruction_object.orig_path),
                                      masked_folder=Path(reconstruction_object.masked_png_path),
                                      mask_folder=Path(reconstruction_object.mask_path),
                                      output_path=Path(reconstruction_object.output_path),
                                      downscale_factor=2,
                                      debug=False)
        ortery.process(image_idx_start=151)

        reconstruction_object.mode = 'up'
        # Compute poses for environment
        gs_reco_up = COLMAPReconstruction(image_path=Path(reconstruction_object.image_masked_path),  # masked_png_path
                                          output_path=Path(reconstruction_object.fused_path),
                                          camera_model=reconstruction_object.camera_model,
                                          database_name='database_0.db',
                                          resize=reconstruction_object.resize,
                                          single_camera=True,  # Do not change this!!!
                                          gpu=True,
                                          colmap_exe=COLMAP_EXE,
                                          magick_exe=MAGICK_EXE
                                          )
        gs_reco_up.run(reference_reconstruction=reconstruction_object.CALIBRATION_OBJECT(
            dataset_path=DATASET_BASE_PATH).reconstruction_path,
                       sparse_id=0,
                       image_list_path=Path(reconstruction_object.output_path) / 'image_list.txt')

        if DEBUG:
            reco_path = reconstruction_object.dataset_path / reconstruction_object.object_name / 'fused'
            project_reference = COLMAPLoader(project_path=reco_path,
                                             img_orig=reco_path / 'images',
                                             sparse_folder_path=(reco_path / 'sparse/0').__str__(),
                                             load_sparse_only=True,
                                             load_depth=False)
            reco_project = project_reference.project_list[0]

            project_vs = ColmapVisualization(reco_project)
            project_vs.visualization(frustum_scale=0.1, image_type='image', show_dense=False, point_size=5)

        # Copy rest of the images to fused folder
        reconstruction_object.mode = 'down'
        shutil.copytree(reconstruction_object.image_masked_path,
                        (Path(reconstruction_object.fused_path) / 'images').__str__(),
                        dirs_exist_ok=True)

        if platform.system() == 'Windows':
            image_list_path = (Path(reconstruction_object.output_path) / 'image_list.txt').__str__().replace('\\up\\',
                                                                                                             '\\down\\'),
        else:
            image_list_path = (Path(reconstruction_object.output_path) / 'image_list.txt').__str__().replace('/up',
                                                                                                             '/down')

        reconstruction_object.mode = 'up'
        gs_reco_up.registrate_images_into_existing_model(
            database_path=(Path(reconstruction_object.fused_path) / 'distorted/database_0.db').__str__(),
            working_dir_images=(Path(reconstruction_object.fused_path) / 'images').__str__(),
            image_list_path=image_list_path,
            sparese_model_path=(Path(reconstruction_object.fused_path) / 'sparse/0').__str__(),
            output_path=(Path(reconstruction_object.fused_path) / 'sparse/0').__str__(),
            image_registration_mapper_settings=reconstruction_object.MATCHING)

    reco_path = reconstruction_object.dataset_path / reconstruction_object.object_name / 'fused'
    # Load Colmap project folder
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
                             show=DEBUG,
                             image_name=os.path.join(reco_path, "{}.png".format(reconstruction_object.object_name)))
    # reco_project.save(data_type='txt', dense=False, appendix='')

    # Convert colmap text to nerf format
    # reco_path = reconstruction_object.dataset_path / reconstruction_object.object_name / 'fused'
    # convert_colmap2nerf(aabb_scale=4,
    #                    image_path=reco_path / 'images',
    #                    colmap_txt_path=(reco_path / 'sparse/0').__str__(),
    #                    output_path=(reco_path / 'sparse/transform.json').__str__())

    # Train gaussian splatting on object
    train_gaussian_splatting_wrapper(data_path=reconstruction_object.fused_path,
                                     model_path=(Path(reconstruction_object.fused_path) / 'gs').__str__(),
                                     TEST_ITERATION=[7_000, 30_000],
                                     SAVE_ITERATION=[7_000, 30_000],
                                     gui=DEBUG
                                     )

    reconstruction_object.mode = 'fused'
    gen = URDFGenerator(object_path=reconstruction_object.gs_o3d_point_cloud_path(iteration=30_000),
                        urdf_template='./src/dataset/urdf_object_template.urdf',
                        meta_info=reconstruction_object,
                        ycb_path=YCB_MODEL_PATH,
                        object_type=reconstruction_object.TYPE)
    gen.generate(obj_path=reconstruction_object.urdf_obj_path,
                 urdf_path=reconstruction_object.urdf_file_path,
                 alpha=reconstruction_object.ALPHA)

    reconstruction_object.gs_cleaning(t=gen.center_translation, R=gen.center_rotation)

    if DEBUG:
        reco_path = reconstruction_object.dataset_path / reconstruction_object.object_name / 'fused'
        # Load Colmap project folder
        project_reference = COLMAPLoader(project_path=reco_path,
                                         img_orig=reco_path / 'images',
                                         sparse_folder_path=(reco_path / 'sparse/0').__str__(),
                                         load_sparse_only=True,
                                         load_depth=False)
        reco_project = project_reference.project_list[0]
        project_vs = ColmapVisualization(reco_project)
        project_vs.visualization(frustum_scale=0.1, image_type='image', show_dense=False, point_size=5)
