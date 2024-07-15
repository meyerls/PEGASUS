# COLMAP Wrapper

<img alt="PyPI" src="https://img.shields.io/pypi/v/colmap-wrapper?label=PyPI"> <a href="https://img.shields.io/pypi/pyversions/colmap-wrapper"><img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/colmap-wrapper"></a> <a href="https://github.com/meyerls/colmap_wrapper/blob/main/LICENSE"><img alt="license" src="https://img.shields.io/github/license/meyerls/colmap-wrapper"></a>

<!--a href="https://github.com/meyerls/colmap-wrapper/actions"><img alt="GitHub Workflow Status" src="https://img.shields.io/github/workflow/status/meyerls/colmap-wrapper/Python%20package"></a-->

## About

Colmap wrapper is a library to work with colmap projects. The purpose is the simplification to read e.g. rgb images,
depth
images, camera poses, sparse point clouds etc. Additionally a visualization for a colmap project is provided.

<p align="center">
    <img width="40%" src="img/img_1.png">
    <img width="40%" src="img/img_2.png">
</p>

## Installation

Make sure that you have a Python version >=3.8 installed.

This repository is tested on Python 3.8+ and can currently only be installed
from [PyPi](https://pypi.org/project/colmap-wrapper/).

 ````bash
pip install dataloader-wrapper
 ````

## Usage

### Single Reconstruction

To visualize a single reconstruction from COLMAP, the following code reads all colmap elements and visualizes them. For
this case an example reconstruction project is provided as shown at the top of the readme.

```python
from colmap_wrapper.dataloader import COLMAPLoader
from colmap_wrapper.visualization import ColmapVisualization
from colmap_wrapper.data.download import Dataset

downloader = Dataset()
downloader.download_bunny_dataset()

project = COLMAPLoader(project_path=downloader.file_path)

colmap_project = project.project

# Acess camera, images and sparse + dense point cloud
camera = colmap_project.cameras
images = colmap_project.images
sparse = colmap_project.get_sparse()
dense = colmap_project.get_dense()

# Visualize COLMAP Reconstruction
project_vs = ColmapVisualization(colmap_project)
project_vs.visualization(frustum_scale=0.7, image_type='image', image_resize=0.4)
```

### Multiple Incomplete Reconstruction

In case of an incomplete reconstruction colmap creates partial reconstructions. In this case a list of
reconstructions can be called as shown below.

```python
from colmap_wrapper.dataloader import COLMAPLoader
from colmap_wrapper.visualization import ColmapVisualization

project = COLMAPLoader(project_path="[PATH2COLMAP_PROJECT]", image_resize=0.3)

# project.projects is a list containing single dataloader projects
for COLMAP_MODEL in project.projects:
    project_vs = ColmapVisualization(colmap=COLMAP_MODEL)
    project_vs.visualization(frustum_scale=0.7, image_type='image')
```

## References

* [PyExifTool](https://github.com/sylikc/pyexiftool): A library to communicate with the [ExifTool](https://exiftool.org)
  command- application. If you have trouble installing it please refer to the PyExifTool-Homepage.

```bash
# For Ubuntu users:
wget https://exiftool.org/Image-ExifTool-12.51.tar.gz
gzip -dc Image-ExifTool-12.51.tar.gz | tar -xf -
cd Image-ExifTool-12.51
perl Makefile.PL
make test
sudo make install
```

* To Visualize the Reconstruction on an OSM-Map the implementation
  from [GPS-visualization-Python](https://github.com/tisljaricleo/GPS-visualization-Python) is used. A guide to
  visualize gps data can be found
  on [Medium](https://towardsdatascience.com/simple-gps-data-visualization-using-python-and-open-street-maps-50f992e9b676)
  .
