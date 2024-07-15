import copy

import pybullet as p
import time
import pybullet_data
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import json
from pathlib import Path
from xml.dom.minidom import parse
from scipy.spatial.transform import Rotation
import random
from typing import Union
import tempfile
import distutils.dir_util


class PybulletEngine:
    def __init__(self, asset_folder: Union[str, list], output_path_json: str = 'simulation_steps.json', simulation_steps: int = 1000,
                 gui: bool = True):
        self.trajectory_path = Path(output_path_json)
        self.trajectory_path.parent.mkdir(exist_ok=True, parents=True)

        self.asset_folder = asset_folder
        if gui:
            self.physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
        else:
            self.physicsClient = p.connect(p.DIRECT)  # or p.DIRECT for non-graphical version
        if isinstance(asset_folder, str):
            p.setAdditionalSearchPath(asset_folder)  # optionally
        elif isinstance(asset_folder, list):

            asset_folder_list = copy.deepcopy(asset_folder)
            tempdir = tempfile.gettempdir()
            self.asset_folder = (Path(tempdir) / 'pegasus_urdf').__str__()

            for asset in asset_folder_list:
                from_dir = asset.__str__()
                to_dir = self.asset_folder
                distutils.dir_util.copy_tree(from_dir, to_dir)

            p.setAdditionalSearchPath(self.asset_folder)
        else:
            raise ValueError("Asset folder must be a string or a list of strings, Currently: {}".format(asset_folder))

        p.setGravity(0, 0, -50)

        if gui:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        self.asset_list = {'environment': {},
                           'object': {}}

        self.simulation_steps = simulation_steps

    def add_object(self, object_instance, start_pos: list = [0, 0, 0], start_orientation_euler: list = [0, 0, 0]):

        name: str = object_instance.urdf_file_name
        type: str = object_instance.TYPE
        class_name: str = object_instance.__class__.__name__
        # start_orientation_quat = p.getQuaternionFromEuler(start_orientation_euler)
        if type == 'environment':
            start_orientation_quat = (0, 0, 0, 1)
        else:
            random.seed(None)
            object_ID: int = object_instance.ID
            start_orientation_quat = (
                random.uniform(0, 1),
                random.uniform(0, 1),
                random.uniform(0, 1),
                random.uniform(0, 1)
            )

        obj_name = name.split('.')[0]

        ID = p.loadURDF(name, start_pos, start_orientation_quat)
        if type == 'environment':
            self.asset_list['environment'].update({obj_name: {'bullet_id': [ID], "class_name": class_name}})
        elif type == 'object':
            if obj_name not in self.asset_list['object'].keys():
                urdf_path = Path(self.asset_folder) / name
                document = parse(urdf_path.__str__())
                center_of_mass_string = document.getElementsByTagName('inertial')[0].getElementsByTagName('origin')[
                    0].getAttribute('xyz').split(' ')
                center_of_mass = [float(center_of_mass_string[0]),
                                  float(center_of_mass_string[1]),
                                  float(center_of_mass_string[2])]

                self.asset_list['object'].update(
                    {obj_name: {'bullet_id': [ID], 'center_of_mass': center_of_mass, "class_name": class_name,
                                "object_ID": object_ID}})
            else:
                self.asset_list['object'][obj_name]['bullet_id'].append(ID)
        else:
            raise ValueError('Wrong entity - {}'.format(type))

    def simulate(self):
        width = 128
        height = 128

        # img = np.random.rand(width, height)
        # img = [tandard_normal((50,100))
        # image = plt.imshow(img, interpolation='none', animated=True, label="blah")
        # ax = plt.gca()

        fov = 80
        aspect = width / height
        near = 0.02
        far = 3

        view_matrix = p.computeViewMatrix([0, 0, 0.2], [0, 0, 0], [0.2, 0, 0])
        projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

        timestep = 1. / 1000.  # 1 / 500
        p.setTimeStep(timestep)

        # time.sleep(30)

        number_of_assets = 1
        for key in self.asset_list['object'].keys():
            number_of_assets += self.asset_list['object'][key]['bullet_id'].__len__()

        P = {key: {} for key in range(number_of_assets)}
        for i in tqdm.tqdm(range(self.simulation_steps)):
            p.stepSimulation()
            # time.sleep(timestep)

            images_open_gl = p.getCameraImage(width,
                                              height,
                                              view_matrix,
                                              projection_matrix,
                                              renderer=p.ER_BULLET_HARDWARE_OPENGL)
            # Plot both images - should show depth values of 0.45 over the cube and 0.5 over the plane
            np_array = images_open_gl[2][..., :3]
            for obj_id in P.keys():
                t, q = p.getBasePositionAndOrientation(obj_id)
                if False:
                    for key in self.asset_list['object'].keys():
                        if obj_id in self.asset_list['object'][key]['id']:
                            t_diff = np.asarray(self.asset_list['object'][key]['center_of_mass'])
                            t -= t_diff
                            # t -= np.asarray([0, 0, 0.07]) # add offest as object would touch ground of gs
                t = np.asarray(t)
                if obj_id > 0:
                    q_mod = Rotation.from_quat(q)  # * Rotation.from_quat((1, 0, 0, 0))
                    q_mod = q_mod.as_quat()
                    q = (q_mod[0], q_mod[1], q_mod[2], q_mod[3])

                # t[0] = -t[0]  # coord conversion

                P[obj_id].update({i: {'t': tuple(t), 'q': q}})

            # print(print(p.getVisualShapeData(1)[0][5]))

            # image.set_data(np.random.rand(width, height))
            # ax.plot([0])
            # plt.draw()
            # plt.show()
            # plt.pause(0.01)
            # image.draw()

        json_file = {
            'asset_infos': self.asset_list,
            'trajectory': P
        }
        with open(self.trajectory_path, 'w') as f:
            json.dump(json_file, f)

        p.disconnect()


if __name__ == '__main__':
    # from src.dataset_objects_old dimport WoodenBlock, CupNoodle01, CupNoodle02, CupNoodle03
    # from src.dataset_envs_old import PlainTableSetup_02, Garden, PlainTableSetup_05

    from src.dataset_objects import *
    from src.dataset.dataset_envs import *

    DATASET_PATH = 'C:\\Users\\meyerls\\Documents\\AIST\\data\\pegasus_dataset'
    URDF_ASSET_FOLDER = 'C:\\Users\\meyerls\\Documents\\AIST\\data\\pegasus\\urdf'

    env = PlainTableSetup(dataset_path=DATASET_PATH)
    # env = MannholeCover(dataset_path=DATASET_PATH)
    obj1 = CrackerBox(dataset_path=DATASET_PATH)
    obj2 = ChocoJello(dataset_path=DATASET_PATH)
    obj3 = RedBowl(dataset_path=DATASET_PATH)
    obj4 = WoodenBlock(dataset_path=DATASET_PATH)
    obj5 = DominoSugar(dataset_path=DATASET_PATH)
    obj6 = YellowMustard(dataset_path=DATASET_PATH)
    obj7 = Banana(dataset_path=DATASET_PATH)
    obj8 = MaxwellCoffee(dataset_path=DATASET_PATH)
    obj9 = RedCup(dataset_path=DATASET_PATH)
    obj10 = Pitcher(dataset_path=DATASET_PATH)
    obj11 = SoftScrub(dataset_path=DATASET_PATH)
    obj12 = TomatoSoup(dataset_path=DATASET_PATH)
    obj13 = Spam(dataset_path=DATASET_PATH)
    obj14 = StrawberryJello(dataset_path=DATASET_PATH)
    obj15 = Tuna(dataset_path=DATASET_PATH)
    obj16 = Pen(dataset_path=DATASET_PATH)
    obj104 = CupNoodle04(dataset_path=DATASET_PATH)

    # np.random.seed(42)

    py_engine = PybulletEngine(asset_folder=URDF_ASSET_FOLDER, gui=True, simulation_steps=4000)
    py_engine.add_object(object_instance=env, start_pos=env.START_POSITION_PYBULLET)

    py_engine.add_object(object_instance=obj104, start_pos=env.define_start_pos())
    if False:
        py_engine.add_object(object_instance=obj2, start_pos=env.define_start_pos())
        py_engine.add_object(object_instance=obj3, start_pos=env.define_start_pos())
        py_engine.add_object(object_instance=obj4, start_pos=env.define_start_pos())
        py_engine.add_object(object_instance=obj5, start_pos=env.define_start_pos())
        py_engine.add_object(object_instance=obj6, start_pos=env.define_start_pos())
        py_engine.add_object(object_instance=obj7, start_pos=env.define_start_pos())
        py_engine.add_object(object_instance=obj8, start_pos=env.define_start_pos())
        py_engine.add_object(object_instance=obj9, start_pos=env.define_start_pos())
        # py_engine.add_object(object_instance=obj10, start_pos=env.define_start_pos())
        py_engine.add_object(object_instance=obj11, start_pos=env.define_start_pos())
        py_engine.add_object(object_instance=obj12, start_pos=env.define_start_pos())
        py_engine.add_object(object_instance=obj13, start_pos=env.define_start_pos())
        py_engine.add_object(object_instance=obj14, start_pos=env.define_start_pos())
        py_engine.add_object(object_instance=obj15, start_pos=env.define_start_pos())
        py_engine.add_object(object_instance=obj16, start_pos=env.define_start_pos())
    py_engine.simulate()
