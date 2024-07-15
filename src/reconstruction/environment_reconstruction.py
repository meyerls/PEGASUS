from src.dataset.dataset_envs import *
from src.dataset.data_sfm_reconstruction import COLMAPReconstruction
from src.dataset.data_alignment import ReconstructionAlignment
from src.gs.gs_training import train_gaussian_splatting_wrapper
from src.dataset.data_urdf import URDFGenerator

import os
import sys
from pathlib import Path
import numpy as np

sys.path.append("./submodules/colmap-wrapper")
from colmap_wrapper.dataloader import COLMAPLoader
from colmap_wrapper.visualization import ColmapVisualization

# Set PYTORCH_CUDA_ALLOC_CONF environment variable
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "caching_allocator"

MAGICK_EXE = "C:\\Program Files\\ImageMagick-7.1.1-Q16-HDRI\\magick.exe"
COLMAP_EXE = "/usr/local/bin/colmap"
DATASET_BASE_PATH = '/media/se86kimy/PortableSSD/data/pegasus'
reco_list = []

reconstruction_object = Garden(dataset_path=DATASET_BASE_PATH)
# reconstruction_object = MannholeCover(dataset_path=DATASET_BASE_PATH)
# reconstruction_object = Cobblestone(dataset_path=DATASET_BASE_PATH)
# reconstruction_object = Counter(dataset_path=DATASET_BASE_PATH)
# reconstruction_object = PlainTableSetup(dataset_path=DATASET_BASE_PATH)
# reconstruction_object = Desk(dataset_path=DATASET_BASE_PATH)
# reconstruction_object = Asphalt(dataset_path=DATASET_BASE_PATH)
# reconstruction_object = Asphalt(dataset_path=DATASET_BASE_PATH)
#reco_list.append(Tiles(dataset_path=DATASET_BASE_PATH))
#reco_list.append(Grass(dataset_path=DATASET_BASE_PATH))
#reco_list.append(Asphalt2(dataset_path=DATASET_BASE_PATH))
#reco_list.append(Tiles2(dataset_path=DATASET_BASE_PATH))
#reco_list.append(Wood(dataset_path=DATASET_BASE_PATH))

reco_list.append(reconstruction_object)

for reconstruction_object in reco_list:

    if True:
        # Compute poses for environment
        gs_reco = COLMAPReconstruction(image_path=Path(reconstruction_object.orig_path),
                                       output_path=Path(reconstruction_object.reconstruction_path),
                                       camera_model=reconstruction_object.camera_model,
                                       resize=reconstruction_object.resize,
                                       gpu=True,
                                       colmap_exe=COLMAP_EXE,
                                       magick_exe=MAGICK_EXE
                                       )
        gs_reco.run()
        if isinstance(reconstruction_object.SCALE, bool) and reconstruction_object.SCALE == True:
            gs_reco.scale_scene(visualize=False)
        elif isinstance(reconstruction_object.SCALE, float):
            gs_reco.scale_scene_by_const(scale=reconstruction_object.SCALE)
        else:
            print('No scaling is applied!')

        # Align environment to xy plane
        reco_align = ReconstructionAlignment(project_path=Path(reconstruction_object.reconstruction_path))
        reco_align.align2plane(plane_size=2.,
                               plane_normal=np.asarray(reconstruction_object.PLANE_NORMAL),
                               debug=False)
        # reco_align.visualize(add_object=[reco_align.plane_mesh], coord_system=True)
        reco_align.save()

    if False:
        reco_path = Path(reconstruction_object.reconstruction_path)
        project_reference = COLMAPLoader(project_path=reco_path,
                                         img_orig=reco_path / 'images',
                                         sparse_folder_path=(reco_path / 'sparse/0').__str__(),
                                         load_sparse_only=True,
                                         load_depth=False)
        reco_project = project_reference.project_list[0]
        project_vs = ColmapVisualization(reco_project)
        project_vs.visualization(frustum_scale=0.1, image_type='image', show_dense=False, point_size=5)

    # Train gaussian splatting on object
    train_gaussian_splatting_wrapper(data_path=reconstruction_object.reconstruction_path,
                                     model_path=reconstruction_object.gs_model_path,
                                     gui=False)

    gen = URDFGenerator(object_path=reconstruction_object.environment_object(),
                        urdf_template='./src/dataset/urdf_object_template.urdf',
                        object_type=reconstruction_object.TYPE,
                        meta_info=reconstruction_object,
                        ycb_path=r"C:\Users\meyerls\Documents\AIST\data\ycbv\models"
                        )
    gen.generate(obj_path=reconstruction_object.urdf_obj_path,
                 urdf_path=reconstruction_object.urdf_file_path,
                 alpha=reconstruction_object.ALPHA)
