import numpy as np
import colorsys
import torch
from typing import Literal


def ndc_to_screen(ndc_x, ndc_y, screen_width, screen_height):
    screen_x = int((ndc_x + 1) * (screen_width / 2))
    screen_y = int((1 - ndc_y) * (screen_height / 2))
    return screen_x, screen_y


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2]])


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def generate_colors(n, mode: Literal['bgr', 'rgb'] = 'bgr'):
    colors = []

    for i in range(n):
        hue = i / n  # Vary the hue based on the index
        saturation = 0.7  # adjust the saturation and lightness as needed
        lightness = 0.6

        rgb_color = colorsys.hls_to_rgb(hue, lightness, saturation)

        if mode == 'bgr':
            b, g, r = rgb_color[2], rgb_color[1], rgb_color[0]
            # Convert RGB to BGR
            bgr_color = (b, g, r)
            colors.append(bgr_color)
        elif mode == 'rgb':
            colors.append(tuple(c for c in rgb_color))
        else:
            raise ValueError("Color mode {} is not supported", mode)

    return torch.asarray(colors, dtype=torch.float32).to('cuda')


def rotate_x(theta):
    """
    Perform a 3D rotation around the X-axis.

    Parameters:
    - theta (float): The angle of rotation in radians (between 0 and 2π).

    Returns:
    - numpy.ndarray: The rotation matrix.
    """
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotation_matrix = np.array([[1, 0, 0],
                               [0, cos_theta, -sin_theta],
                               [0, sin_theta, cos_theta]])
    return rotation_matrix

def rotate_y(theta):
    """
    Perform a 3D rotation around the Y-axis.

    Parameters:
    - theta (float): The angle of rotation in radians (between 0 and 2π).

    Returns:
    - numpy.ndarray: The rotation matrix.
    """
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotation_matrix = np.array([[cos_theta, 0, sin_theta],
                               [0, 1, 0],
                               [-sin_theta, 0, cos_theta]])
    return rotation_matrix

def rotate_z(theta):
    """
    Perform a 3D rotation around the Z-axis.

    Parameters:
    - theta (float): The angle of rotation in radians (between 0 and 2π).

    Returns:
    - numpy.ndarray: The rotation matrix.
    """
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotation_matrix = np.array([[cos_theta, -sin_theta, 0],
                               [sin_theta, cos_theta, 0],
                               [0, 0, 1]])
    return rotation_matrix