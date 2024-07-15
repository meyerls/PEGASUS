import copy
import os
import sys

sys.path.append("./submodules/gaussian-splatting-pegasus")
sys.path.append("./submodules/colmap-wrapper")

from argparse import ArgumentParser

import tqdm
import numpy as np
import pylab as plt
import torch
import threading
from pathlib import Path
import warnings
# Own
from colmap_wrapper.dataloader import (read_images_binary, read_cameras_binary)

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render, network_gui
from utils.sh_utils import RGB2SH
from utils.general_utils import safe_state

from src.dataset.ycb_objects import *
from src.dataset.cup_noodle_dataset import *
from src.dataset.dataset_envs import *
from src.engine.physical_simulation import PybulletEngine
from src.utility.graphic_utils import *
from src.gs.pegasus_setup import PegasusSetup

from src.gs.render import render_rgb_and_depth, render_visib_mask, render_silhouette_mask, \
    render_semanticsegmentation_mask


class PEGASUS(object):
    GUI_NETWORKING_ACTIVATED: bool = False
    IP: str = "127.0.0.1"
    PORT: int = 6009
    LOAD_ITERATION: int = 30_000
    SH_DEGREE: int = 3

    def __init__(self,
                 dataset_path: str,
                 env_dataset_path: str,
                 urdf_asset_folder: Union[str, list],
                 gs_env_list: list,
                 gs_object_list: list,
                 mode: Literal["dynamic", "static"] = 'static',
                 camera_trajectory_mode: Literal["random", "sequence", 'random+zoom'] = 'random',
                 render_height: int = 480,
                 render_width: int = 640,
                 num_cameras: int = 1,
                 simulation_steps: int = 100,
                 num_camera_interpolation_steps: int = 1,
                 QUIET: bool = False,  # What is this for?
                 publish2gui: bool = False
                 ):
        self.URDF_ASSET_FOLDER = urdf_asset_folder
        # Set up command line argument parser
        self.parser = ArgumentParser(description="Testing script parameters")
        self.model = ModelParams(self.parser, sentinel=True)
        self.pipeline = PipelineParams(self.parser)

        self.dataset_path = dataset_path
        if env_dataset_path:
            self.env_dataset_path = env_dataset_path
        else:
            self.env_dataset_path = dataset_path
        self.render_height = render_height
        self.render_width = render_width

        self.dataset_base_path = './dataset'
        self.num_cameras = num_cameras
        self.num_camera_interpolation_steps = num_camera_interpolation_steps

        self.fps = 50
        self.QUIET = QUIET
        self.GUI = publish2gui
        self.mode = mode
        self.simulation_steps = simulation_steps
        self.camera_trajectory_mode = camera_trajectory_mode

        if publish2gui and not self.GUI_NETWORKING_ACTIVATED:
            network_gui.init(self.IP, self.PORT)
            self.GUI_NETWORKING_ACTIVATED = True

        # Preload gaussian splatting point clouds
        with (torch.no_grad()):
            self.gaussian_environment_pre_load = {}
            for env_idx in range(0, len(gs_env_list)):
                gaussian_environment = GaussianModel(self.SH_DEGREE)
                gaussian_environment.meta_info = gs_env_list[env_idx]
                gaussian_environment.load_ply(gs_env_list[env_idx].gaussian_point_cloud_path(self.LOAD_ITERATION))

                # load colmap data
                cam_extr = read_images_binary(Path(gs_env_list[env_idx].reconstruction_path) / 'sparse/0/images.bin')
                cam_intr = read_cameras_binary(Path(gs_env_list[env_idx].reconstruction_path) / 'sparse/0/cameras.bin')

                self.gaussian_environment_pre_load.update(
                    {
                        gs_env_list[env_idx].object_name:
                            {
                                'gs': gaussian_environment,
                                'cam_extr': cam_extr,
                                'cam_intr': cam_intr,
                            }
                    })

            self.gaussian_object_pre_load = {}
            for obj_idx in range(0, len(gs_object_list)):
                gs_object_list[obj_idx].mode = 'fused'
                gaussian_object = GaussianModel(self.SH_DEGREE)
                gaussian_object.load_ply(
                    gs_object_list[obj_idx].gaussian_point_cloud_path(iteration=self.LOAD_ITERATION))
                gaussian_object.meta_info = gs_object_list[obj_idx]
                self.gaussian_object_pre_load.update({gs_object_list[obj_idx].object_name: gaussian_object})

    def init(self, dataset_name, scene_id):

        self.dataset_name = dataset_name
        self.scene_id = scene_id

        # Init all params for gaussian splatting (trajectory, videos)
        self.pegasus_setup = PegasusSetup(pybullet_trajectory_path=self.physics_file,
                                          dataset_path=self.dataset_path,
                                          env_dataset_path=self.env_dataset_path,
                                          render_height=self.render_height,
                                          render_width=self.render_width,
                                          mode=self.mode)

        self.pegasus_setup.cam_extr = self.gaussian_environment_pre_load[self.selected_env_name]['cam_extr']
        self.pegasus_setup.cam_intr = self.gaussian_environment_pre_load[self.selected_env_name]['cam_intr']

        # Setup bop dataset writer
        self.pegasus_dataset = PegasusBOPDatasetWriter(dataset_name=dataset_name,
                                                       dataset_output_path=Path(self.dataset_base_path),
                                                       camera_intr=self.pegasus_setup.cam_intr,
                                                       render_width=self.pegasus_setup.render_width,
                                                       render_height=self.pegasus_setup.render_height,
                                                       object_models=self.pegasus_setup.object_data.keys(),
                                                       object_dataset_path=self.dataset_path,
                                                       scene_id=scene_id)

        self.viewport_cam_list = self.pegasus_setup.create_camera_trajectory(num_cameras=self.num_cameras,
                                                                             num_interpolation_steps=self.num_camera_interpolation_steps,
                                                                             mode=self.camera_trajectory_mode)
        self.pegasus_setup.init_video_streams(
            output=self.pegasus_dataset.dataset_path / 'video/{:06d}'.format(scene_id), fps=self.fps)

        # hacky way of saving path to model pcd
        sys.argv.append('-m')
        sys.argv.append(self.pegasus_setup.environment.gs_model_path)
        self.args = get_combined_args(self.parser)

        print("Rendering Environment" + self.pegasus_setup.environment.reconstruction_path)

        # Initialize system state (RNG)
        safe_state(self.QUIET)
        self.pipe = self.pipeline.extract(self.args)

        dataset = self.model.extract(self.args)
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    def init_bullet(self,
                    env_list: list,
                    obj_list: list,
                    dataset_name: str,
                    scene_id: int,
                    min_num_objects: int = 1,
                    max_num_objects: int = 1,
                    random: bool = True):
        engine_path = Path('src/tools/') / self.dataset_base_path / dataset_name

        self.py_engine = PybulletEngine(asset_folder=URDF_ASSET_FOLDER,
                                        output_path_json=(engine_path / 'engine/{:06d}_simulation_steps.json'.format(
                                            scene_id)).__str__(),
                                        simulation_steps=self.simulation_steps,
                                        gui=self.GUI)

        if not random:
            np.random.seed(42)
        else:
            np.random.seed(None)

        self.physics_file = self.py_engine.trajectory_path

        if min_num_objects > obj_list.__len__():
            min_num_objects = obj_list.__len__()
            warnings.warn(
                "Number of min objects selected is larger than parsed objects. Set number of objects to lowest possible.")

        if max_num_objects > obj_list.__len__():
            max_num_objects = obj_list.__len__()
            warnings.warn(
                "Number of objects selected is lower than parsed objects. Set number of objects to lowest possible.")

        # Selected random environment (only 1 env)
        select_env = env_list[np.random.randint(0, env_list.__len__())]
        self.selected_env_name = select_env.object_name
        # Number of objects
        random_num_objects = np.random.randint(min_num_objects, max_num_objects + 1)
        # Indecies for selected objects
        # random_objects_idx = np.unique(np.random.randint(0, obj_list.__len__(), num_objects)).tolist()
        random_objects_idx = np.random.choice(range(obj_list.__len__()), random_num_objects, replace=False).tolist()

        print('Env: {}. Selected {} objects.'.format(select_env.__class__.__name__, random_objects_idx.__len__()))

        random_objects = [obj_list[i] for i in random_objects_idx]

        # Add selected envs and objects into pybullet engine
        self.py_engine.add_object(object_instance=select_env, start_pos=select_env.START_POSITION_PYBULLET)
        for obj in random_objects:
            self.py_engine.add_object(object_instance=obj, start_pos=select_env.define_start_pos())
        self.py_engine.simulate()

    def init_start_position(self):
        self.semantic_colors = generate_colors(
            self.pegasus_setup.object_data.__len__())  # TOdo: this should be only executed once!

        gaussians_object_list = {}
        for object_name in self.pegasus_setup.object_data.keys():
            for id in self.pegasus_setup.object_data[object_name]['bullet_id']:
                gs_object = self.gaussian_object_pre_load[object_name]

                gs_object._features_dc_color = copy.deepcopy(gs_object._features_dc)
                gs_object._features_rest_color = copy.deepcopy(gs_object._features_rest)

                gs_semeantic_color = self.semantic_colors[id - 1]
                gs_object._features_dc_semantics = RGB2SH(gs_semeantic_color)
                gs_object._features_rest_semantics = torch.asarray([0, 0, 0])

                gaussians_object_list.update({id: copy.deepcopy(gs_object)})

        # Static
        if self.pegasus_setup.mode == 'static':
            self.current_gaussians_object_list = self.pegasus_setup.static_object_pose(
                gaussians_object_list=gaussians_object_list)
        # Dynamic
        elif self.pegasus_setup.mode == 'dynamic':
            self.current_gaussians_object_list = self.pegasus_setup.dynamic_object_pose(
                gaussians_object_list=gaussians_object_list)
        else:
            raise ValueError('Mode -{}- not available'.format(self.pegasus_setup.mode))

    def generate_dataset(self, data_points: list, save_bop: bool = True, save_video: bool = True):
        with (torch.no_grad()):
            if self.GUI:
                if network_gui.conn is None:
                    network_gui.try_connect()

            bar = tqdm.tqdm(total=len(self.viewport_cam_list))
            for i in range(len(self.viewport_cam_list)):
                gaussian_scene = copy.deepcopy(
                    list(self.gaussian_environment_pre_load[self.selected_env_name].values())[0])
                gaussian_environment = list(self.gaussian_environment_pre_load[self.selected_env_name].values())[0]

                # Compose scene for rgb rendering
                for gs_object_id in self.current_gaussians_object_list.keys():
                    curr_object = self.current_gaussians_object_list[gs_object_id]
                    curr_object._features_dc = copy.deepcopy(curr_object._features_dc_color)
                    curr_object._features_rest = copy.deepcopy(curr_object._features_rest_color)
                    gaussian_scene.merge_gaussians(gaussian=curr_object)

                if self.GUI:
                    try:
                        net_image_bytes = None
                        custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                        if custom_cam is not None:
                            net_image = render(custom_cam, gaussian_scene, pipe, background, scaling_modifer)["render"]
                            net_image_bytes = memoryview(
                                (torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2,
                                                                                              0).contiguous().cpu().numpy())
                        network_gui.send(net_image_bytes, dataset.source_path)

                    except Exception as e:
                        network_gui.conn = None
                        # print("Termination error: ", e)

                if i < len(self.viewport_cam_list):
                    # Set camera from camera trajectory
                    viewpoint_cam = self.viewport_cam_list[i]

                    rgb_image = None
                    depth_image = None
                    mask_silhouette = None
                    individual_seg_masks = None
                    seg_image = None
                    semantic_segmentation_mask = None
                    object_center_image = None

                    if 'rgb' in data_points:
                        # Render rgb
                        rgb_image, depth_image = render_rgb_and_depth(cam=viewpoint_cam,
                                                                      gs_scene=gaussian_scene,
                                                                      pipe_settings=self.pipe,
                                                                      bg=self.background,
                                                                      debug=False)

                    if 'seg_sil' in data_points:
                        # Render silhouette mask
                        mask_silhouette = render_silhouette_mask(cam=viewpoint_cam,
                                                                 gs_object_list=self.current_gaussians_object_list,
                                                                 gs_env=gaussian_environment,
                                                                 width=self.pegasus_setup.render_width,
                                                                 height=self.pegasus_setup.render_height,
                                                                 color_set=self.semantic_colors,
                                                                 pipe_settings=self.pipe,
                                                                 bg=self.background)

                    if 'seg_vis' in data_points:
                        # Render visible mask
                        individual_seg_masks, seg_image = render_visib_mask(cam=viewpoint_cam,
                                                                            gs_environment=gaussian_environment,
                                                                            gs_object_list=self.current_gaussians_object_list,
                                                                            color_set=self.semantic_colors,
                                                                            width=self.pegasus_setup.render_width,
                                                                            height=self.pegasus_setup.render_height,
                                                                            pipe_settings=self.pipe,
                                                                            bg=self.background)

                    if 'sem_seg' in data_points:
                        semantic_segmentation_mask = render_semanticsegmentation_mask(cam=viewpoint_cam,
                                                                                      gs_environment=gaussian_environment,
                                                                                      gs_object_list=self.current_gaussians_object_list,
                                                                                      color_set=self.semantic_colors,
                                                                                      width=self.pegasus_setup.render_width,
                                                                                      height=self.pegasus_setup.render_height,
                                                                                      pipe_settings=self.pipe,
                                                                                      bg=self.background,
                                                                                      debug=False)
                    self.pegasus_dataset.add_scene_camera_json(frame_id=i)

                    if save_bop:
                        if False:
                            self.pegasus_dataset.write_training_data(
                                rgb_image=(np.ascontiguousarray(rgb_image) * 255).astype('uint8'),
                                seg_image=individual_seg_masks,
                                semantic_masks=semantic_segmentation_mask,
                                mask_silhouette=mask_silhouette,
                                depth_image=(depth_image.numpy() * 1000).astype(np.uint16),
                                frame_id=i)  # meters in millimeter

                        if True:
                            threading.Thread(target=write_training_data,
                                             args=((np.ascontiguousarray(rgb_image) * 255).astype('uint8'),
                                                   self.pegasus_dataset.rgb_path,
                                                   individual_seg_masks,
                                                   self.pegasus_dataset.mask_visib_path,
                                                   mask_silhouette,
                                                   self.pegasus_dataset.mask_path,
                                                   semantic_segmentation_mask,
                                                   self.pegasus_dataset.sem_mask_path,
                                                   (depth_image.numpy() * 1000).astype(np.uint16),
                                                   self.pegasus_dataset.depth_path,
                                                   i),
                                             ).start()

                        self.pegasus_dataset.add_scene_gt_json(time_step=i,
                                                               gs_object_list=self.current_gaussians_object_list,
                                                               cam=viewpoint_cam,
                                                               rgb_image=(np.ascontiguousarray(rgb_image) * 255).astype(
                                                                   'uint8'),
                                                               debug=False)

                    object_center_image = self.pegasus_setup.draw_object_center(
                        image=(np.ascontiguousarray(rgb_image) * 255).astype('uint8'),
                        gaussians_object_list=self.current_gaussians_object_list,
                        camera=viewpoint_cam,
                        semantic_colors=self.semantic_colors,
                        K=self.pegasus_dataset.K)
                    if False:
                        # imageio.imwrite('image_object_pose.png', object_center_image)
                        plt.imshow(object_center_image)
                        plt.show()

                    if save_video:
                        # Save images to video stream
                        self.pegasus_setup.write_image2video(
                            rgb=(np.ascontiguousarray(rgb_image) * 255).astype('uint8'),
                            depth=depth_image,
                            seg=seg_image,
                            center_image=object_center_image)
                    bar.update(1)

                    if self.pegasus_setup.mode == 'dynamic':
                        self.current_gaussians_object_list = self.pegasus_setup.update_object_pose(
                            gaussians_object_list=self.current_gaussians_object_list,
                            timestep=i + 1)

    def save2bop(self):
        self.pegasus_setup.close_video_streams()
        self.pegasus_dataset.write_scene_camera_json()
        self.pegasus_dataset.write_scene_gt_json()
        print('Saved BOP data')


if __name__ == '__main__':
    DATASET_PATH = '/home/se86kimy/Documents/data/RamenDataset'
    PEGASET_PATH = '/home/se86kimy/Documents/data/PEGASET'

    ENV_DATASET_PATH = '/home/se86kimy/Documents/data/RamenDataset'
    URDF_ASSET_FOLDER = ['/home/se86kimy/Documents/data/RamenDataset/urdf',
                         '/home/se86kimy/Documents/data/PEGASET/urdf']

    os.environ['PEGASUS_PATH'] = os.path.join(os.path.abspath(os.path.curdir), "dataset")
    from src.dataset.data_writer import PegasusBOPDatasetWriter, write_training_data, write_models, \
        convert_scenewise_to_imagewise_ndds, calculate_gt_info

    env1 = MannholeCover(dataset_path=ENV_DATASET_PATH)
    env2 = Cobblestone(dataset_path=ENV_DATASET_PATH)
    env3 = Asphalt(dataset_path=ENV_DATASET_PATH)
    env4 = Tiles(dataset_path=ENV_DATASET_PATH)
    env5 = Grass(dataset_path=ENV_DATASET_PATH)
    env6 = Asphalt2(dataset_path=DATASET_PATH)
    env7 = Tiles2(dataset_path=DATASET_PATH)
    env8 = Asphalt2(dataset_path=DATASET_PATH)
    env9 = Wood(dataset_path=DATASET_PATH)

    obj1 = CrackerBox(dataset_path=PEGASET_PATH)
    obj2 = ChocoJello(dataset_path=PEGASET_PATH)
    obj3 = RedBowl(dataset_path=PEGASET_PATH)
    obj4 = WoodenBlock(dataset_path=PEGASET_PATH)
    obj5 = DominoSugar(dataset_path=PEGASET_PATH)
    obj6 = YellowMustard(dataset_path=PEGASET_PATH)
    obj7 = Banana(dataset_path=PEGASET_PATH)
    obj8 = MaxwellCoffee(dataset_path=PEGASET_PATH)
    obj9 = RedCup(dataset_path=PEGASET_PATH)
    obj10 = Pitcher(dataset_path=PEGASET_PATH)
    obj11 = SoftScrub(dataset_path=PEGASET_PATH)
    obj12 = TomatoSoup(dataset_path=PEGASET_PATH)
    obj13 = Spam(dataset_path=PEGASET_PATH)
    obj14 = StrawberryJello(dataset_path=PEGASET_PATH)
    obj15 = Tuna(dataset_path=PEGASET_PATH)
    # obj16 = Drill(dataset_path=DATASET_PATH)
    obj17 = Pen(dataset_path=PEGASET_PATH) # Black (y)
    obj18 = Scissors(dataset_path=PEGASET_PATH)  # Black (y)
    obj19 = SmallClamp(dataset_path=PEGASET_PATH)  # Black
    obj20 = LargeClamp(dataset_path=PEGASET_PATH)  # Black
    obj21 = FoamBrick(dataset_path=PEGASET_PATH)

    # Ramen Dataset
    obj101 = CupNoodle01(dataset_path=DATASET_PATH)
    obj102 = CupNoodle02(dataset_path=DATASET_PATH)
    obj103 = CupNoodle03(dataset_path=DATASET_PATH)
    obj104 = CupNoodle04(dataset_path=DATASET_PATH)
    obj105 = CupNoodle05(dataset_path=DATASET_PATH)
    obj106 = CupNoodle06(dataset_path=DATASET_PATH)
    obj107 = CupNoodle07(dataset_path=DATASET_PATH)
    obj108 = CupNoodle08(dataset_path=DATASET_PATH)
    obj109 = CupNoodle09(dataset_path=DATASET_PATH)
    obj110 = CupNoodle10(dataset_path=DATASET_PATH)
    obj111 = CupNoodle11(dataset_path=DATASET_PATH)
    obj112 = CupNoodle12(dataset_path=DATASET_PATH)
    obj113 = CupNoodle13(dataset_path=DATASET_PATH)
    obj114 = CupNoodle14(dataset_path=DATASET_PATH)
    obj115 = CupNoodle15(dataset_path=DATASET_PATH)
    obj116 = CupNoodle16(dataset_path=DATASET_PATH)
    obj117 = CupNoodle17(dataset_path=DATASET_PATH)
    obj118 = CupNoodle18(dataset_path=DATASET_PATH)
    obj119 = CupNoodle19(dataset_path=DATASET_PATH)
    obj120 = CupNoodle20(dataset_path=DATASET_PATH)
    obj121 = CupNoodle21(dataset_path=DATASET_PATH)
    obj122 = CupNoodle22(dataset_path=DATASET_PATH)
    obj123 = CupNoodle23(dataset_path=DATASET_PATH)
    obj124 = CupNoodle24(dataset_path=DATASET_PATH)
    obj125 = CupNoodle25(dataset_path=DATASET_PATH)
    obj126 = CupNoodle26(dataset_path=DATASET_PATH)
    obj127 = CupNoodle27(dataset_path=DATASET_PATH)
    obj128 = CupNoodle28(dataset_path=DATASET_PATH)
    obj129 = CupNoodle29(dataset_path=DATASET_PATH)
    obj130 = CupNoodle30(dataset_path=DATASET_PATH)


    obj_list = [obj17, obj18, obj19, obj20]

    env_list = [
        env1, env2, env3, env4, env5, env6, env7, env8, env9
    ]

    dataset_path_folder = './dataset/'
    dataset_name = 'pegasus_ycb_test'
    mode = "dynamic"  # 'static'
    GUI = False
    num_scenes = 10
    min_num_objects = 3
    max_num_objects = 6
    image_height = 480
    image_width = 640
    render_data_points = ['rgb', 'depth', 'seg_vis', 'seg_sil', 'sem_seg']  # ['rgb', 'depth'],
    convert_from_scenewise2imagewise = True

    pegasus = PEGASUS(dataset_path=PEGASET_PATH,
                      env_dataset_path=ENV_DATASET_PATH,
                      urdf_asset_folder=URDF_ASSET_FOLDER,
                      gs_env_list=env_list,
                      gs_object_list=obj_list,
                      render_height=image_height,
                      render_width=image_width,
                      simulation_steps=310,
                      num_cameras=10,
                      num_camera_interpolation_steps=30,
                      publish2gui=GUI,
                      QUIET=False,
                      mode="dynamic",  # static
                      camera_trajectory_mode="random",
                      )

    write_models(dataset_path=PEGASET_PATH,
                 object_list=obj_list,
                 model_path=(Path(dataset_path_folder) / dataset_name / 'models').__str__())

    for scene_id in range(1, num_scenes + 1):
        pegasus.init_bullet(env_list=env_list,
                            obj_list=obj_list,
                            min_num_objects=min_num_objects,
                            max_num_objects=max_num_objects,
                            dataset_name=dataset_name,
                            scene_id=scene_id)

        pegasus.init(dataset_name=dataset_name,
                     scene_id=scene_id)

        pegasus.init_start_position()

        pegasus.generate_dataset(data_points=render_data_points,
                                 save_video=True,
                                 save_bop=True)

        pegasus.save2bop()

        del pegasus.py_engine

    if convert_from_scenewise2imagewise:
        calculate_gt_info(dataset_name=dataset_name, num_scenes=num_scenes, object_list=[obj.ID for obj in obj_list])

        all_scene_ids = list(range(1, num_scenes))
        train_ids = list(range(1, int(np.round(0.8 * all_scene_ids.__len__()))))
        test_ids = list(range(int(np.round(0.8 * all_scene_ids.__len__())), num_scenes))

        train_string = ""
        for num in train_ids:
            train_string = train_string + str(num) + ","

        convert_scenewise_to_imagewise_ndds(input_path=pegasus.pegasus_dataset.train_data_path.__str__(),
                                            output_path=(
                                                    pegasus.pegasus_dataset.train_data_path.parent / 'train_ndds').__str__(),
                                            scene_ids_process=train_string[:-1])
        test_string = ""
        for num in test_ids:
            test_string = test_string + str(num) + ","

        convert_scenewise_to_imagewise_ndds(input_path=pegasus.pegasus_dataset.train_data_path.__str__(),
                                            output_path=(
                                                    pegasus.pegasus_dataset.train_data_path.parent / 'test_ndds').__str__(),
                                            scene_ids_process=test_string[:-1])
