import numpy as np
import open3d as o3d


def points2pcd(points: np.ndarray) -> o3d.geometry.PointCloud:
    """
    Converts a numpy array to an Open3D point cloud.

    This function converts a numpy array representing the XYZ coordinates of points to an Open3D point cloud. It assigns a
    uniform blue color to the point cloud.

    Args:
        points (numpy.ndarray): Array of shape Nx3 with XYZ locations of points.

    Returns:
        open3d.geometry.PointCloud: An Open3D point cloud object with the converted points and blue color.

    """

    colors = [[0, 0, 1] for i in range(points.shape[0])]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def kabsch_umeyama(pointset_A, pointset_B) -> tuple:
    """
    Applies the Kabsch-Umeyama algorithm to align two sets of points and compute the optimal rotation, scaling, and translation.

    The Kabsch-Umeyama algorithm finds the optimal transformation by minimizing the root-mean-square deviation (RMSD) between
    corresponding points in the two sets. It returns the rotation matrix, scaling factor, and translation vector that aligns
    pointset_A to pointset_B.

    Args:
        pointset_A (numpy.ndarray): Array of a set of points in n-dim.
        pointset_B (numpy.ndarray): Array of a set of points in n-dim.

    Returns:
        Tuple[numpy.ndarray, float, numpy.ndarray]: The rotation matrix (3x3), scaling factor (scalar), and translation vector (3x1).

    Raises:
        AssertionError: If the shapes of pointset_A and pointset_B do not match.

    Reference:
        Source and Explanation: https://zpl.fi/aligning-point-patterns-with-kabsch-umeyama-algorithm/

    """
    assert pointset_A.shape == pointset_B.shape
    n, m = pointset_A.shape

    # Find centroids of both point sets
    EA = np.mean(pointset_A, axis=0)
    EB = np.mean(pointset_B, axis=0)

    VarA = np.mean(np.linalg.norm(pointset_A - EA, axis=1) ** 2)

    # Covariance matrix
    H = ((pointset_A - EA).T @ (pointset_B - EB)) / n

    # SVD H = UDV^T
    U, D, VT = np.linalg.svd(H)

    # Detect and prevent reflection
    d = np.sign(np.linalg.det(U) * np.linalg.det(VT))
    S = np.diag([1] * (m - 1) + [d])

    # rotation, scaling and translation
    R = U @ S @ VT
    c = VarA / np.trace(np.diag(D) @ S)
    t = EA - c * R @ EB

    return R, c, t


def convert_to_cartesian(lat: float, lon: float, elevation: float) -> tuple:
    """
    Converts latitude, longitude, and elevation to Cartesian coordinates.

    Args:
        lat (float): Latitude in degrees.
        lon (float): Longitude in degrees.
        elevation (float): Elevation in meters.

    Returns:
        Tuple[float, float, float]: Cartesian coordinates (x, y, z).

    Note:
        - This function assumes WGS84 ellipsoid and uses the radius of the Earth plus elevation to calculate Cartesian coordinates.

    Reference:
        https://itecnote.com/tecnote/python-how-to-convert-longitudelatitude-elevation-to-cartesian-coordinates/

    """
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    R = 6378137.0 + elevation  # radius of the earth

    x = R * np.cos(lat) * np.cos(lon)
    y = R * np.cos(lat) * np.sin(lon)
    z = R * np.sin(lat)

    return x, y, z
