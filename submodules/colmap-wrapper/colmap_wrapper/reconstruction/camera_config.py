"""
Copyright (c) 2023 Lukas Meyer
Licensed under the MIT License.
See LICENSE file for more information.
"""

# Built-in/Generic Imports
# ...

# Libs
try:
    import pycolmap
except ModuleNotFoundError:
    print("Pycolmap is not used!")


# Own modules
# ...


class CameraConfig(object):
    def __init__(self):
        self.image_size: tuple
        self.model: str
        self.camera_params: list
        self.camera: pycolmap.Camera


class UnknownCamera(CameraConfig):
    def __init__(self):
        CameraConfig.__init__(self)
        self.image_size = (3200, 3200)
        self.model = 'SIMPLE_PINHOLE'
        self.camera_params = []

        self.camera = pycolmap.Camera(
            model=self.model,
            width=self.image_size[0],
            height=self.image_size[1],
            params=self.camera_params,
        )


class P1Camera(CameraConfig):
    def __init__(self):
        CameraConfig.__init__(self)
        self.image_size = (8192, 5460)
        self.model = 'SIMPLE_PINHOLE'
        self.camera_params = []

        self.camera = pycolmap.Camera(
            model=self.model,
            width=self.image_size[0],
            height=self.image_size[1],
            params=self.camera_params,
        )


class DSLRCamera(CameraConfig):
    def __init__(self):
        CameraConfig.__init__(self)

        self.image_size = (6000, 4000)
        self.model = 'OPENCV'
        self.camera_params = [4518.9, 4511.7, 3032.2, 2020.9, -0.1623, 0.0902, 0, 0]

        self.camera = pycolmap.Camera(
            model='OPENCV',
            width=self.image_size[0],
            height=self.image_size[1],
            params=self.camera_params,
        )