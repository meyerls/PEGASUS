#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2022 Lukas Meyer
Licensed under the MIT License.
See LICENSE file for more information.

Code for COLMAP readout borrowed from https://github.com/uzh-rpg/colmap_utils/tree/97603b0d352df4e0da87e3ce822a9704ac437933
"""

# Built-in/Generic Imports
from pathlib import Path
from typing import Union

# Libs
import numpy as np

# Own modules
from colmap_wrapper.dataloader.project import COLMAPProject
from colmap_wrapper.gps.registration import GPSRegistration


class COLMAPLoader(GPSRegistration):
    def __init__(self, project_path: Union[str, Path],
                 dense_pc: str = 'fused.ply',
                 bg_color: np.ndarray = np.asarray([1, 1, 1]),
                 exif_read: bool = False,
                 img_orig: Union[str, Path, None] = None,
                 output_status_function=None,
                 oriented: bool = False,
                 sparse_folder_path: Union[str, Path, bool] = False,
                 load_sparse_only: bool = False,
                 load_depth: bool = True
                 ):
        """
        Constructor for COLMAPLoader class.

        Args:
            project_path (Union[str, Path]): Path to the COLMAP project.
            dense_pc (str): Name of the dense point cloud file.
            bg_color (np.ndarray): Background color as a NumPy array.
            exif_read (bool): Flag to indicate whether to read EXIF data.
            img_orig (Union[str, Path, None]): Path to the original image.
            output_status_function: Function for displaying status output.
            oriented (bool): Flag to indicate whether the project is oriented.
        """

        GPSRegistration.__init__(self)

        self.exif_read = exif_read
        self.vis_bg_color = bg_color
        self._project_path: Path = Path(project_path)
        # Original image path. Colmap does not copy exif data. Therfore we have to access original data
        self._img_orig_path: Union[Path, None] = Path(img_orig) if img_orig else None

        if '~' in str(self._project_path):
            self._project_path = self._project_path.expanduser()

        # If sparse folder is parsed as an argument
        if sparse_folder_path:
            self._sparse_base_path = Path(sparse_folder_path)
        else:
            self._sparse_base_path = self._project_path.joinpath('sparse')
        if not self._sparse_base_path.exists():
            raise ValueError('Colmap project structure: sparse folder (cameras, images, points3d) can not be found')

        project_structure = {}
        self._dense_base_path = self._project_path.joinpath('dense')

        # Test if path is a file to get number of subprojects. Single Project with no numeric folder
        if not all([path.is_dir() for path in self._sparse_base_path.iterdir()]):
            project_structure.update({0: {
                "project_path": self._project_path,
                "sparse": self._sparse_base_path,
                "dense": self._dense_base_path}})
        else:
            # In case of folder with reconstruction number after sparse (multiple projects) (e.g. 0,1,2)
            # WARNING: dense folder 0 an 1 are not the same as sparse 0 and 1 (ToDO: find out if this is always the case)
            # for project_index, sparse_project_path in enumerate(list(self._sparse_base_path.iterdir())):
            #    project_structure.update({project_index: {"sparse": sparse_project_path}})

            for project_index, sparse_project_path in enumerate(list(self._dense_base_path.iterdir())):
                project_structure.update({project_index: {"sparse": sparse_project_path.joinpath('sparse')}})

            for project_index, dense_project_path in enumerate(list(self._dense_base_path.iterdir())):
                project_structure[project_index].update({"dense": dense_project_path})

            for project_index in project_structure.keys():
                project_structure[project_index].update({"project_path": self._project_path})

        self.project_list = []
        self.model_ids = []

        # n_cores = 1# cpu_count()
        # executor = ThreadPoolExecutor(max_workers=n_cores)

        for project_index in project_structure.keys():
            self.model_ids.append(project_index)

            #       def run():
            project = COLMAPProject(project_path=project_structure[project_index],
                                    project_index=project_index,
                                    dense_pc=dense_pc,
                                    bg_color=bg_color,
                                    exif_read=exif_read,
                                    img_orig=self._img_orig_path,
                                    output_status_function=output_status_function,
                                    oriented=oriented,
                                    load_sparse_only=load_sparse_only,
                                    load_depth=load_depth)

            self.project_list.append(project)

            # executor.submit(run)

        # executor.shutdown(wait=True)

    @property
    def projects(self):
        """
        Get the list of projects.

        Returns:
            List: List of COLMAPProject objects.
        """
        return self.project_list

    @projects.setter
    def projects(self, projects):
        self.project_list = projects

    def fuse_projects(self):
        """
        Fuse the projects.

        Returns:
            NotImplementedError: The function is not implemented yet.
        """
        return NotImplementedError

    def save_project(self, output_path):
        """
         Save the project.

         Args:
             output_path: The output path to save the project.

         Returns:
             NotImplementedError: The function is not implemented yet.
         """
        return NotImplementedError


if __name__ == '__main__':
    from colmap_wrapper.visualization import ColmapVisualization
    from colmap_wrapper import USER_NAME

    MODE = 'multi'

    if MODE == "single":
        from colmap_wrapper.data.download import Dataset

        downloader = Dataset()
        downloader.download_bunny_dataset()

        project = COLMAPLoader(project_path=downloader.file_path)

        colmap_project = project.project

        camera = colmap_project.cameras
        images = colmap_project.images
        sparse = colmap_project.get_sparse()
        dense = colmap_project.get_dense()

        project_vs = ColmapVisualization(colmap=colmap_project, image_resize=0.4)
        project_vs.visualization(frustum_scale=0.8, image_type='image')
    elif MODE == "multi":
        project = COLMAPLoader(project_path='/home/{}/Dropbox/07_data/For5G/22_11_14/reco'.format(USER_NAME),
                               dense_pc='fused.ply')

        for model_idx, COLMAP_MODEL in enumerate(project.projects):
            camera = COLMAP_MODEL.cameras
            images = COLMAP_MODEL.images
            sparse = COLMAP_MODEL.get_sparse()
            dense = COLMAP_MODEL.get_dense()
            project_vs = ColmapVisualization(colmap=COLMAP_MODEL, image_resize=0.4)
            project_vs.visualization(frustum_scale=0.8, image_type='image')
