import json
import math
import os
import random

import numpy as np
from scipy.spatial.transform import Rotation as R
from shapely.geometry import Polygon


def quaternion_to_rotation_matrix(r):
    """
    Parameters:
    r : np.ndarray
        An [N, 4] array of quaternions (w, x, y, z).

    Returns:
    rotation_matrices : np.ndarray
        An [N, 3, 3] array of rotation matrices.
    """
    norm = np.sqrt(r[:, 0] ** 2 + r[:, 1] ** 2 + r[:, 2] ** 2 + r[:, 3] ** 2)

    q = r / norm[:, None]

    R = np.zeros((q.shape[0], 3, 3), dtype=np.float32)

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def rotation_matrix_to_quaternion(rotation_matrices):
    """
    Convert a batch of rotation matrices into quaternions.

    Parameters:
    rotation_matrices : np.ndarray
        An [N, 3, 3] array of rotation matrices.

    Returns:
    quaternions : np.ndarray
        An [N, 4] array of quaternions (w, x, y, z).
    """
    r = R.from_matrix(rotation_matrices)
    return r.as_quat()[:, (3, 0, 1, 2)]


def z_rotation_matrix(theta_deg):
    theta_rad = math.radians(theta_deg)
    return np.array([
        [np.cos(theta_rad), -np.sin(theta_rad), 0],
        [np.sin(theta_rad), np.cos(theta_rad), 0],
        [0, 0, 1]
    ])


def strip_symmetric(matrix):
    triangle = np.zeros((matrix.shape[0], 6), dtype=np.float32)

    triangle[:, 0] = matrix[:, 0, 0]
    triangle[:, 1] = matrix[:, 0, 1]
    triangle[:, 2] = matrix[:, 0, 2]
    triangle[:, 3] = matrix[:, 1, 1]
    triangle[:, 4] = matrix[:, 1, 2]
    triangle[:, 5] = matrix[:, 2, 2]
    return triangle


def scale_rotation_matrix(scale, rotation):
    rot_matrix = quaternion_to_rotation_matrix(rotation)

    scale_matrix = np.zeros((scale.shape[0], 3, 3), dtype=np.float32)
    for i in range(3):
        scale_matrix[:, i, i] = scale[:, i]

    scale_rot_matrix = np.einsum('bij,bjk->bik', rot_matrix, scale_matrix)
    return scale_rot_matrix


def covariance_matrix(scaling, rotation):
    scale_rot_matrix = scale_rotation_matrix(scaling, rotation)
    return np.einsum('bij,bkj->bik', scale_rot_matrix, scale_rot_matrix.transpose(0, 2, 1))


def minkowski_sum(poly1, poly2):
    """
    Minkowski sum approximation
    """
    result = []
    for p1 in poly1.exterior.coords:
        for p2 in poly2.exterior.coords:
            result.append((p1[0] + p2[0], p1[1] + p2[1]))
    return Polygon(result).convex_hull


def find_random_point_on_contour(contour):
    length = contour.length
    distance = random.uniform(0, length)
    point = contour.interpolate(distance)
    return point


def compute_aabb_diagonal(aabb):
    """
    Compute the diagonal of the AABB.

    Parameters:
    aabb : np.ndarray
        A [2, 2] array representing the AABB with min and max corners.

    Returns:
    float
        The length of the diagonal of the AABB.
    """
    min_corner = aabb[0]
    max_corner = aabb[1]
    diagonal_vector = max_corner - min_corner
    diagonal_length = np.linalg.norm(diagonal_vector)
    return diagonal_length


def split_into_random_subsets(values, min_subset_size, max_subset_size):
    random.shuffle(values)

    subsets = []
    i = 0

    while i < len(values):
        remaining = len(values) - i
        if min_subset_size <= remaining <= max_subset_size:
            subset_size = remaining
        else:
            subset_size = random.randint(min_subset_size, min(max_subset_size, remaining - min_subset_size))

        subsets.append(values[i:i + subset_size])
        i += subset_size

    return subsets


def create_random_sample_indices(data_count, max_sample_count):
    if data_count > max_sample_count:
        return np.random.choice(data_count, max_sample_count)
    else:
        return range(data_count)


def read_split_file(data_path, split_filename):
    split_file = os.path.join(data_path, split_filename)
    with open(split_file, 'r') as file:
        return [os.path.realpath(os.path.join(data_path, line.strip())) for line in file.readlines()]


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __add__(self, other):
        return dotdict({**self, **other})


def load_json(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return None


def save_json(filename, progress):
    with open(filename, 'w') as f:
        json.dump(progress, f)


def farthest_point_sampling(xyz, num_samples):
    num_points = xyz.shape[0]
    indices = np.zeros(num_samples, dtype=int)
    distances = np.full(num_points, np.inf)

    # Randomly select the initial point
    indices[0] = np.random.randint(num_points)
    current_point = xyz[indices[0]]

    for i in range(1, num_samples):
        # Update distances based on the current point
        current_distances = np.sum((xyz - current_point) ** 2, axis=1)
        distances = np.minimum(distances, current_distances)

        # Select the farthest point
        indices[i] = np.argmax(distances)
        current_point = xyz[indices[i]]

    return indices
