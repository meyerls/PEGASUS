from src.dataset.dataset_envs import *
from src.dataset.data_sfm_reconstruction import COLMAPReconstruction
from src.dataset.data_alignment import ReconstructionAlignment
from src.gs.gs_training import train_gaussian_splatting_wrapper
from src.dataset.data_urdf import URDFGenerator

import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import shutil
from PIL import Image
from tqdm.contrib import tzip

# sys.path.append("C:/Users/meyerls/Documents/AIST/code/gaussian-splatting/aruco-estimator")
sys.path.append("./submodules/colmap-wrapper")

# from aruco_estimator.aruco_scale_factor import ArucoScaleFactor
# from aruco_estimator.visualization import ArucoVisualization

from colmap_wrapper.dataloader import COLMAPLoader
from colmap_wrapper.visualization import ColmapVisualization

MAGICK_EXE = "C:\\Program Files\\ImageMagick-7.1.1-Q16-HDRI\\magick.exe"
COLMAP_EXE = "/usr/local/bin/colmap"
# DATASET_BASE_PATH = 'C:\\Users\\meyerls\\Documents\\AIST\\data\\pegasus_dataset'
DATASET_BASE_PATH = './workspace'

# from src.dataset_objects_old import *
from src.dataset.ycb_objects import *
from src.dataset.in_the_wild_dataset import *


class ImageProcessor(object):
    def __init__(self, orig_folder: Path,
                 mask_folder: Path,
                 output_path: Path,
                 orig_extension: str = 'jpg',
                 masked_folder_extension: str = 'png',
                 mask_folder_extension: str = 'png',
                 downscale_factor: float = -1.,
                 debug: bool = False):

        # Base folder for original, masked and mask image
        self.orig_folder: Path = orig_folder
        self.mask_folder: Path = mask_folder

        # Output folder to save mask and masked image
        self.output_path: Path = output_path

        # assume that all images are sorted by name!
        self.all_orig_files: list = sorted(list(self.orig_folder.glob("*.{}".format(orig_extension))))
        self.all_mask_files: list = sorted(list(self.mask_folder.glob("*.{}".format(mask_folder_extension))))

        self.downscale_factor = downscale_factor
        self.debug: bool = debug

        self.output_path_masked: Path = self.output_path / ('image_masked')
        self.output_path_masked.mkdir(exist_ok=True)

    def process(self, image_idx_start=1, skip_images=1):

        image_names_list = []

        # if list(self.output_path_masked.glob("*")).__len__() == self.all_orig_files.__len__():
        #    return 0

        image_idx = 0

        for image_file_path, mask_file_path in tzip(self.all_orig_files, self.all_mask_files):
            if image_idx % skip_images != 0:
                image_idx += 1
                continue
            image_idx += 1

            image_idx = int(mask_file_path.parts[-1].split(".")[0]) + (image_idx_start - 1)
            # image_idx_str = '{:03d}'.format(image_idx)
            image_idx_str = str(image_idx).zfill(image_file_path.name.split(".")[0].__len__())

            image_names_list.append(image_idx_str + '.jpg')

            output_name_masked = self.output_path_masked / (image_idx_str + '.jpg')

            # shutil.copy(masked_file_path, output_name_masked)
            if not output_name_masked.exists():
                mask_image = Image.open(mask_file_path)
                image = Image.open(image_file_path)

                image_np = np.asarray(image).astype(int)
                mask_np = np.asarray(mask_image)

                image_np[mask_np == 0] = 0

                masked_image = Image.fromarray(image_np.astype(np.uint8))
                if self.downscale_factor != -1.:
                    masked_image = masked_image.resize((int(masked_image.width / self.downscale_factor),
                                                        int(masked_image.height / self.downscale_factor)),
                                                       Image.Resampling.LANCZOS)
                masked_image.save(output_name_masked)
            # if not output_name_mask.exists():
            #    shutil.copy(mask_file_path, output_name_mask)

        # print("Masked images, mask have the same size as the original images. Copy to target folder...")

        with open(self.output_path_masked.parent / 'image_list.txt', 'w+') as fp:
            for item in image_names_list:
                # write each item on a new line
                fp.write("%s\n" % item)
            print('Done')

        return image_names_list.__len__()


DEBUG = False
reco_list = []

reco_list.append(Bouillon(dataset_path=DATASET_BASE_PATH))

for reconstruction_object in reco_list:

    reconstruction_object.mode = 'up'  # Set by default
    processor = ImageProcessor(orig_folder=Path(reconstruction_object.orig_path),
                               mask_folder=Path(reconstruction_object.mask_path),
                               output_path=Path(reconstruction_object.output_path),
                               downscale_factor=1,
                               debug=False)
    num_images = processor.process(image_idx_start=1, skip_images=2)

    reconstruction_object.mode = 'down'  # Set by default
    processor = ImageProcessor(orig_folder=Path(reconstruction_object.orig_path),
                               mask_folder=Path(reconstruction_object.mask_path),
                               output_path=Path(reconstruction_object.output_path),
                               downscale_factor=1,
                               debug=False)
    processor.process(image_idx_start=num_images + 1, skip_images=1)

    if True:
        reconstruction_object.mode = 'up'  # Set by default
        #shutil.copytree(reconstruction_object.image_masked_path,
        #                (Path(reconstruction_object.fused_path) / 'images').__str__(),
        #                dirs_exist_ok=True)

        # Compute poses for environment
        gs_reco_up = COLMAPReconstruction(image_path=Path(reconstruction_object.image_masked_path),
                                          output_path=Path(reconstruction_object.fused_path),
                                          camera_model=reconstruction_object.camera_model,
                                          resize=reconstruction_object.resize,
                                          single_camera=True,  # Do not change this!!!
                                          gpu=True,
                                          colmap_exe=COLMAP_EXE,
                                          magick_exe=MAGICK_EXE
                                          )
        gs_reco_up.run()

        if isinstance(reconstruction_object.SCALE, bool) and reconstruction_object.SCALE == True:
            gs_reco_up.scale_scene(aruco_scale=reconstruction_object.ARUCO_SIZE,
                                   img_orig=reconstruction_object.orig_path,
                                   visualize=False)
        elif isinstance(reconstruction_object.SCALE, float):
            gs_reco_up.scale_scene_by_const(scale=reconstruction_object.SCALE)
        else:
            print('No scaling is applied!')

        # Align environment to xy plane
        reco_align = ReconstructionAlignment(project_path=Path(reconstruction_object.fused_path))
        reco_align.align2plane(plane_size=2.,
                               plane_normal=np.asarray(reconstruction_object.PLANE_NORMAL),
                               debug=False)
        # reco_align.visualize(add_object=[reco_align.plane_mesh], coord_system=True)
        reco_align.save()
#
        # Copy rest of the images to fused folder
        reconstruction_object.mode = 'down'
        shutil.copytree(reconstruction_object.image_masked_path,
                        (Path(reconstruction_object.fused_path) / 'images').__str__(),
                        dirs_exist_ok=True)
#
#
        gs_reco_up.registrate_images_into_existing_model(
            database_path=(Path(reconstruction_object.fused_path) / 'distorted/database.db').__str__(),
            working_dir_images=(Path(reconstruction_object.fused_path) / 'images').__str__(),
            image_list_path=(Path(reconstruction_object.output_path) / 'image_list.txt').__str__().replace('\\up\\',
                                                                                                           '\\down\\'),
            sparese_model_path=(Path(reconstruction_object.fused_path) / 'sparse/0').__str__(),
            output_path=(Path(reconstruction_object.fused_path) / 'sparse/0').__str__(),
            image_registration_mapper_settings=reconstruction_object.MATCHING)

    if False:
        reco_path = Path(reconstruction_object.fused_path)
        project_reference = COLMAPLoader(project_path=reco_path,
                                         img_orig=reco_path / 'images',
                                         sparse_folder_path=(reco_path / 'sparse/0').__str__(),
                                         load_sparse_only=True,
                                         load_depth=False)
        reco_project = project_reference.project_list[0]
        project_vs = ColmapVisualization(reco_project)
        project_vs.visualization(frustum_scale=0.01, image_type='image', show_dense=False, point_size=1)

    # Train gaussian splatting on object
    # Train gaussian splatting on object
    train_gaussian_splatting_wrapper(data_path=reconstruction_object.fused_path,
                                     model_path=(Path(reconstruction_object.fused_path) / 'gs').__str__(),
                                     TEST_ITERATION=[7_000, 30_000],
                                     SAVE_ITERATION=[7_000, 30_000],
                                     gui=False
                                     )
    reconstruction_object.mode = 'fused'
    gen = URDFGenerator(object_path=reconstruction_object.gs_o3d_point_cloud_path(iteration=30_000),
                        urdf_template='./src/dataset/urdf_object_template.urdf',
                        object_type=reconstruction_object.TYPE,
                        meta_info=reconstruction_object,
                        ycb_path=reconstruction_object.REFERENCE_DATASET_PATH
                        )
    gen.generate(obj_path=reconstruction_object.urdf_obj_path,
                 urdf_path=reconstruction_object.urdf_file_path,
                 alpha=reconstruction_object.ALPHA)

    reconstruction_object.gs_cleaning(t=gen.center_translation, R=gen.center_rotation)
