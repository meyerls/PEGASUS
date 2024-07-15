#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 21:54:18 2019

@author: Matthieu Zins
"""

import numpy as np
from scipy.spatial.transform import Rotation as Rot

"""
        Contains functions to convert poses between homogeneous matrix 
        representation and quaternion representation or interpolate 
        between two poses using SLERP for the rotation and linear
        interpolation for the position.
"""


def pose_matrix_to_quat(pose):
    """
        Convert a pose from 4x4 matrix to
        a 7-vector (qx, qy, qz, qw, x, y, z)
    """
    assert pose.shape == (4, 4)
    q = Rot.from_matrix(pose[:3, :3]).as_quat()
    return np.hstack((q, pose[:3, 3]))


def pose_quat_to_matrix(pose):
    """
        Convert a pose from 7-vector
        (qx, qy, qz, qw, x, y, z) to a matrix
    """
    assert pose.size == 7
    R = Rot.from_quat(pose[:4]).as_matrix()
    p = np.eye(4, dtype=np.float32)
    p[:3, :3] = R
    p[:3, 3] = pose[4:]
    return p


def apply_pose(pose, pts):
    """
        Apply a rigid transform to points. The points
        have to be Nx3. The transformed points are returned (Nx3)
    """
    assert pts.shape[1] == 3
    if pose.shape == (4, 4) or pose.shape == (3, 4):
        return (pose[:3, :3] @ pts.T + pose[:3, 3].reshape((3, 1))).T
        pass
    elif pose.size == 7:
        return Rot.from_quat(pose[:4]).apply(pts) + pose[4:]
    else:
        raise RuntimeError("invalid pose")


def quaternion_SLERP_interpolate(q1, q2, alpha):
    """
        Interpolate between two quaternions
        with SLERP
    """
    assert alpha >= 0.0 and alpha <= 1.0
    q1 = np.asarray(q1)
    q2 = np.asarray(q2)
    dot = q1.dot(q2)

    if dot < 0:
        q1 = -q1
        dot = -dot

    dot_threshold = 0.9995
    if dot > dot_threshold:
        res = q1 + alpha * (q2 - q1)
        res /= np.linalg.norm(res)
        return res
    theta_0 = np.arccos(dot)
    theta = theta_0 * alpha
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)

    s1 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s2 = sin_theta / sin_theta_0
    return s1 * q1 + s2 * q2


def interpolate_pose(t, t1, pose1, t2, pose2):
    """
        Interpolate the between two poses.
        Linear for the position and spherical
        linear for the rotation (SLERP)
    """
    if pose1.shape == (4, 4):
        pose1 = pose_matrix_to_quat(pose1)
    if pose2.shape == (4, 4):
        pose2 = pose_matrix_to_quat(pose2)

    assert t >= t1 and t <= t2
    t = float(t)
    t1 = float(t1)
    t2 = float(t2)
    r = (t - t1) / (t2 - t1)

    pos = pose1[4:] + r * (pose2[4:] - pose1[4:])
    rot = quaternion_SLERP_interpolate(pose1[:4], pose2[:4], r)

    return pose_quat_to_matrix(np.hstack((rot, pos)))