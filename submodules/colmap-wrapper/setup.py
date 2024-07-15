#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2022 Lukas Meyer
Licensed under the MIT License.
See LICENSE file for more information.
"""

# Built-in/Generic Imports
import setuptools
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'Readme.md'), encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name='colmap_wrapper',
    version='1.1.5',
    description='COLMAP Wrapper',
    license="MIT",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Lukas Meyer',
    author_email='lukas.meyer@fau.de',
    url="https://github.com/meyerls/colmap-wrapper",
    packages=['colmap_wrapper.dataloader',
              'colmap_wrapper.visualization',
              'colmap_wrapper.data',
              'colmap_wrapper.gps',
              'colmap_wrapper.reconstruction'],
    install_requires=["numpy",
                      "pyexiftool",
                      "open3d",
                      "pyquaternion",
                      "matplotlib",
                      "tqdm",
                      "wget",
                      #"pycolmap"
                      ],  # external packages as dependencies
    classifiers=[
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
