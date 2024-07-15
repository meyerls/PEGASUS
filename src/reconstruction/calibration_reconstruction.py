from src.dataset.data_alignment import ReconstructionAlignment
from src.dataset.data_ortery_preperation import OrteryImageProcessor
from src.dataset.data_sfm_reconstruction import COLMAPReconstruction
from src.dataset_objects import CalibrationBoard
from src.gs.gs_training import train_gaussian_splatting_wrapper
from src.dataset.data_urdf import URDFGenerator

from pathlib import Path
import numpy as np

MAGICK_EXE = "C:\\Program Files\\ImageMagick-7.1.1-Q16-HDRI\\magick.exe"
COLMAP_EXE = "C:\\Users\\meyerls\\Documents\\AIST\\misc\\COLMAP-3.8-windows-cuda\\COLMAP.bat"
DATASET_BASE_PATH_OLD = 'C:\\Users\\meyerls\\Documents\\AIST\\data\\pegasus_dataset'
DATASET_BASE_PATH = 'C:\\Users\\meyerls\\Documents\\AIST\\data\\pegasus'

# Load calibration object
reconstruction_object = CalibrationBoard(dataset_path=DATASET_BASE_PATH)
#reconstruction_object = WoodenCalibrationBoard(dataset_path=DATASET_BASE_PATH)
#reconstruction_object = SecurityCalibrationBoard(dataset_path=DATASET_BASE_PATH)

if True:
    # Preprocessing images
    ortery = OrteryImageProcessor(orig_folder=Path(reconstruction_object.orig_path),
                                  masked_folder=Path(reconstruction_object.masked_png_path),
                                  mask_folder=Path(reconstruction_object.mask_path),
                                  output_path=Path(reconstruction_object.output_path),
                                  downscale_factor=2,
                                  debug=False)
    ortery.process()

# Compute poses for environment
gs_reco = COLMAPReconstruction(image_path=Path(reconstruction_object.image_masked_path),  # masked_png_path
                               output_path=Path(reconstruction_object.reconstruction_path),
                               camera_model=reconstruction_object.camera_model,
                               resize=reconstruction_object.resize,
                               single_camera=True, # Do not change this!!!
                               gpu=True,
                               colmap_exe=COLMAP_EXE,
                               magick_exe=MAGICK_EXE
                               )
gs_reco.run(image_list_path=Path(reconstruction_object.output_path) / 'image_list.txt')
gs_reco.scale_scene(visualize=False)


# Align environment to xy plane
reco_align = ReconstructionAlignment(project_path=Path(reconstruction_object.reconstruction_path))
reco_align.align2plane(debug=False, plane_normal=np.asarray([0, 0, 1]))
reco_align.visualize(add_object=[reco_align.plane_mesh], coord_system=True)
reco_align.save()

