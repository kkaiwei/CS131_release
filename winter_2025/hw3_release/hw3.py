from __future__ import print_function
import random
import numpy as np
import time
from PIL import Image
from skimage import color, io
import matplotlib.pyplot as plt
from typing import Tuple
from skimage import filters
from scipy.ndimage import convolve
from scipy.spatial.distance import cdist
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def camera_from_world_transform(d: float = 1.0) -> np.ndarray:
    """Define a transformation matrix in homogeneous coordinates that
    transforms coordinates from world space to camera space, according
    to the coordinate systems in Question 1.


    Args:
        d (float, optional): Total distance of displacement between world and camera
            origins. Will always be greater than or equal to zero. Defaults to 1.0.

    Returns:
        T (np.ndarray): Left-hand transformation matrix, such that c = Tw
            for world coordinate w and camera coordinate c as column vectors.
            Shape = (4,4) where 4 means 3D+1 for homogeneous.
    """
    T = np.eye(4)

    # YOUR CODE HERE
    theta = 3 * np.pi / 4 
    R = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0,             1, 0            ],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

    c_w = np.array([d / np.sqrt(2), 0, d / np.sqrt(2)])
    t = -R @ c_w

    T[:3, :3] = R
    T[:3, 3] = t
    # END YOUR CODE

    assert T.shape == (4, 4)
    return T

def apply_transform(T: np.ndarray, points: np.ndarray) -> Tuple[np.ndarray]:
    """Apply a transformation matrix to a set of points.

    Hint: You'll want to first convert all of the points to homogeneous coordinates.
    Each point in the (3,N) shape edges is a length 3 vector for x, y, and z, so
    appending a 1 after z to each point will make this homogeneous coordinates.

    You shouldn't need any loops for this function.

    Args:
        T (np.ndarray):
            Left-hand transformation matrix, such that c = Tw
                for world coordinate w and camera coordinate c as column vectors.
            Shape = (4,4) where 4 means 3D+1 for homogeneous.
        points (np.ndarray):
            Shape = (3,N) where 3 means 3D and N is the number of points to transform.

    Returns:
        points_transformed (np.ndarray):
            Transformed points.
            Shape = (3,N) where 3 means 3D and N is the number of points.
    """
    N = points.shape[1]
    assert points.shape == (3, N)

    # You'll replace this!
    points_transformed = np.zeros((3, N))

    # YOUR CODE HERE
    points_homogeneous = np.vstack((points, np.ones((1, N))))  # Shape (4, N)
    points_transformed_homogeneous = T @ points_homogeneous  # Shape (4, N)
    points_transformed = points_transformed_homogeneous[:3, :]
    # END YOUR CODE

    assert points_transformed.shape == (3, N)
    return points_transformed


def intersection_from_lines(
    a_0: np.ndarray, a_1: np.ndarray, b_0: np.ndarray, b_1: np.ndarray
) -> np.ndarray:
    """Find the intersection of two lines (infinite length), each defined by a
    pair of points.

    Args:
        a_0 (np.ndarray): First point of first line; shape `(2,)`.
        a_1 (np.ndarray): Second point of first line; shape `(2,)`.
        b_0 (np.ndarray): First point of second line; shape `(2,)`.
        b_1 (np.ndarray): Second point of second line; shape `(2,)`.

    Returns:
        np.ndarray:
    """
    # Validate inputs
    assert a_0.shape == a_1.shape == b_0.shape == b_1.shape == (2,)
    assert a_0.dtype == a_1.dtype == b_0.dtype == b_1.dtype == float

    # Intersection point between lines
    out = np.zeros(2)

    # YOUR CODE HERE
    a0 = np.append(a_0, 1)
    a1 = np.append(a_1, 1)
    b0 = np.append(b_0, 1)
    b1 = np.append(b_1, 1)

    la = np.cross(a0, a1)
    lb = np.cross(b0, b1)

    intersection_homogeneous = np.cross(la, lb)
    out = intersection_homogeneous[:2] / intersection_homogeneous[2]
    # END YOUR CODE

    assert out.shape == (2,)
    assert out.dtype == float

    return out


def optical_center_from_vanishing_points(
    v0: np.ndarray, v1: np.ndarray, v2: np.ndarray
) -> np.ndarray:
    """Compute the optical center of our camera intrinsics from three vanishing
    points corresponding to mutually orthogonal directions.

    Hints:
    - Your `intersection_from_lines()` implementation might be helpful here.
    - It might be worth reviewing vector projection with dot products.

    Args:
        v0 (np.ndarray): Vanishing point in image space; shape `(2,)`.
        v1 (np.ndarray): Vanishing point in image space; shape `(2,)`.
        v2 (np.ndarray): Vanishing point in image space; shape `(2,)`.

    Returns:
        np.ndarray: Optical center; shape `(2,)`.
    """
    assert v0.shape == v1.shape == v2.shape == (2,), "Wrong shape!"

    optical_center = np.zeros(2)

    # YOUR CODE HERE
    side1 = v1 - v0 
    side2 = v2 - v1
    
    # The altitude from v2 perpendicular to side1 (v0v1)
    altitude1 = np.array([-side1[1], side1[0]])  # Perpendicular to v0v1
    line1 = np.cross(np.append(v2, 1), np.append(v2 + altitude1, 1))
    
    # The altitude from v0 perpendicular to side2 (v1v2)
    altitude2 = np.array([-side2[1], side2[0]])  # Perpendicular to v1v2
    line2 = np.cross(np.append(v0, 1), np.append(v0 + altitude2, 1))
    
    intersection = np.cross(line1, line2)
    optical_center = intersection[:2] / intersection[2]
    # END YOUR CODE

    assert optical_center.shape == (2,)
    return optical_center

def focal_length_from_two_vanishing_points(
    v0: np.ndarray, v1: np.ndarray, optical_center: np.ndarray
) -> np.ndarray:
    """Compute focal length of camera, from two vanishing points and the
    calibrated optical center.

    Args:
        v0 (np.ndarray): Vanishing point in image space; shape `(2,)`.
        v1 (np.ndarray): Vanishing point in image space; shape `(2,)`.
        optical_center (np.ndarray): Calibrated optical center; shape `(2,)`.

    Returns:
        float: Calibrated focal length.
    """
    assert v0.shape == v1.shape == optical_center.shape == (2,), "Wrong shape!"

    f = None

    # YOUR CODE HERE
    v0_shifted = v0 - optical_center
    v1_shifted = v1 - optical_center

    dot_product_xy = v0_shifted[0] * v1_shifted[0] + v0_shifted[1] * v1_shifted[1]
    f = np.sqrt(-dot_product_xy)
    # END YOUR CODE

    return float(f)

def physical_focal_length_from_calibration(
    f: float, sensor_diagonal_mm: float, image_diagonal_pixels: float
) -> float:
    """Compute the physical focal length of our camera, in millimeters.

    Args:
        f (float): Calibrated focal length, using pixel units.
        sensor_diagonal_mm (float): Length across the diagonal of our camera
            sensor, in millimeters.
        image_diagonal_pixels (float): Length across the diagonal of the
            calibration image, in pixels.

    Returns:
        float: Calibrated focal length, in millimeters.
    """
    f_mm = None

    # YOUR CODE HERE
    pixels_to_mm = sensor_diagonal_mm / image_diagonal_pixels
    f_mm = f * pixels_to_mm
    # END YOUR CODE

    return f_mm


