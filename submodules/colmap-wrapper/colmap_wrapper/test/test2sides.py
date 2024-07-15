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
from colmap_wrapper.colmap.colmap import COLMAP
from colmap_wrapper.visualization import ColmapVisualization
from colmap_wrapper import USER_NAME

if __name__ == '__main__':
    project = COLMAP(project_path='/media/{}/Samsung_T5/For5G/reco/23_03_17/03'.format(USER_NAME), dense_pc='fused.ply')

    colmap_project = project.project

    camera = colmap_project.cameras
    images = colmap_project.images
    sparse = colmap_project.get_sparse()
    dense = colmap_project.get_dense()

    for COLMAP_MODEL in project.projects:
        project_vs = ColmapVisualization(colmap=COLMAP_MODEL, bg_color=np.asarray([0, 0, 0]))
        project_vs.visualization(frustum_scale=0.4, image_type='image', point_size=0.001)

    print('Finished')
