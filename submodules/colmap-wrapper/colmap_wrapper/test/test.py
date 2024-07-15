#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2022 Lukas Meyer
Licensed under the MIT License.
See LICENSE file for more information.
"""
import numpy as np

# Own Modules
from colmap_wrapper.data.download import Dataset
from colmap_wrapper.dataloader import COLMAPLoader
from colmap_wrapper.visualization import ColmapVisualization

if __name__ == '__main__':
    downloader = Dataset()
    downloader.download_bunny_reco_dataset()

    project = COLMAPLoader(project_path=downloader.file_path)

    colmap_project = project.project

    camera = colmap_project.cameras
    images = colmap_project.images
    sparse = colmap_project.get_sparse()
    dense = colmap_project.get_dense()

    # project_vs = ColmapVisualization(dataloader=colmap_project, bg_color=np.asarray([0, 0, 0]))
    # project_vs.visualization(frustum_scale=0.4, image_type='image', point_size=0.001)

    print('Finished')
