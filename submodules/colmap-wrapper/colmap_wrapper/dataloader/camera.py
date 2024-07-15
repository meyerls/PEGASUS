#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2022 Lukas Meyer
Licensed under the MIT License.
See LICENSE file for more information.
"""

# Built-in/Generic Imports
import collections

import numpy as np
from PIL import Image

from colmap_wrapper.dataloader import (qvec2rotmat, rotmat2qvec)

CameraModel = collections.namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])

CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS])

CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model)
                           for camera_model in CAMERA_MODELS])

class Point3D:
    def __init__(self,
                 point_id: int,
                 xyz: np.ndarray,
                 rgb: np.ndarray,
                 error: float,
                 image_ids: np.ndarray,
                 point2D_idxs: np.ndarray):
        self.id = point_id
        self.xyz = xyz
        self.rgb = rgb
        self.error = error
        self.image_ids = image_ids
        self.point2D_idxs = point2D_idxs


class ImageInformation(object):
    def __init__(self,
                 image_id: int,
                 qvec: np.ndarray,
                 tvec: np.ndarray,
                 camera_id: int,
                 image_name: str,
                 xys: np.ndarray,
                 point3D_ids: np.ndarray,
                 point3DiD_to_kpidx: dict):
        """
        @param image_id: number of image
        @param qvec: quaternion of the camera viewing direction
        @param tvec: translation of the image/camera position
        @param camera_id: camera id
        @param image_name: name of th image
        @param xys:
        @param point3D_ids:
        @param point3DiD_to_kpidx:

        """
        # Parsed arguments
        self.id = image_id
        self.qvec = qvec
        self.tvec = tvec
        self.camera_id = camera_id
        self.name = image_name
        self.xys = xys
        self.point3D_ids = point3D_ids
        self.point3DiD_to_kpidx = point3DiD_to_kpidx

        self.intrinsics = None
        self.extrinsics = np.eye(4)

        self.path = None
        self.depth_image_geometric_path = None
        self.depth_image_photometric_path = None

        self.__image = None

        self.set_extrinsics()

    @property
    def image(self):
        return self.getData(1.0)

    @image.setter
    def image(self, image: np.ndarray):
        self.__image = image

    def getData(self, downsample: float = 1.0) -> np.ndarray:
        if self.__image is None:
            try:
                with Image.open(self.path) as img:
                    width, height = img.size
                    img = img.resize((int(width * downsample), int(height * downsample)))
                return np.asarray(img).astype(np.uint8)
            except FileNotFoundError:
                img = np.zeros((400, 400))

                return np.asarray(img).astype(np.uint8)

        return self.__image

    @property
    def depth_image_geometric(self):
        if self.depth_image_geometric_path == None:
            return None
        from colmap_wrapper.colmap import read_array
        return read_array(path=self.depth_image_geometric_path)

    @property
    def depth_image_photometric(self):
        if self.depth_image_photometric_path == None:
            return None
        from colmap_wrapper.colmap import read_array
        return read_array(path=self.depth_image_photometric_path)

        #    @property
        #    def extrinsics(self) -> np.ndarray:
        #        Rwc = self.Rwc()
        #        twc = self.twc()
        #
        #        M = np.eye(4)
        #        M[:3, :3] = Rwc
        #        M[:3, 3] = twc

        # return M

    def set_extrinsics(self, T: [None, np.ndarray] = None):
        if isinstance(T, type(None)):
            Rwc = self.Rwc()
            twc = self.twc()
        else:
            Rwc = T[:3, :3]
            twc = T[:3, 3]

        self.extrinsics = np.eye(4)
        self.extrinsics[:3, :3] = Rwc
        self.extrinsics[:3, 3] = twc

    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)

    def rotmat2qvec(self):
        return rotmat2qvec(self.extrinsics[:3, :3])

    def qtvec(self):
        return self.qvec.ravel().tolist() + self.tvec.ravel().tolist()

    def Rwc(self):
        return self.qvec2rotmat().transpose()

    def twc(self):
        return np.dot(-self.qvec2rotmat().transpose(), self.tvec)

    def Rcw(self):
        return self.qvec2rotmat()

    def tcw(self):
        return self.tvec

    def Twc(self):
        Twc = np.eye(4)
        Twc[0:3, 3] = self.twc()
        Twc[0:3, 0:3] = self.Rwc()

        return Twc

    def Tcw(self):
        Tcw = np.eye(4)
        Tcw[0:3, 3] = self.tcw()
        Tcw[0:3, 0:3] = self.Rcw()

        return Tcw


class Intrinsics:
    def __init__(self, camera):
        self._cx = None
        self._cy = None
        self._fx = None
        self._fy = None

        self.camera = camera
        self.load_from_colmap(camera=self.camera)

    def load_from_colmap(self, camera):
        self.fx = camera.params[0]
        self.fy = camera.params[1]
        self.cx = camera.params[2]
        self.cy = camera.params[3]
        self.k1 = camera.params[4]

    @property
    def cx(self):
        return self._cx

    @cx.setter
    def cx(self, cx):
        self._cx = cx

    @property
    def cy(self):
        return self._cy

    @cy.setter
    def cy(self, cy):
        self._cy = cy

    @property
    def fx(self):
        return self._fx

    @fx.setter
    def fx(self, fx):
        self._fx = fx

    @property
    def fy(self):
        return self._fy

    @fy.setter
    def fy(self, fy):
        self._fy = fy

    @property
    def K(self):
        K = np.asarray([[self.fx, 0, self.cx],
                        [0, self.fy, self.cy],
                        [0, 0, 1]])

        return K
