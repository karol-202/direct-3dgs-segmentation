import numpy as np

from data_utils.extra_feature import FEATURE_ROTATION_QUAT, FEATURE_SCALE, FEATURE_OPACITY, FEATURE_REST, FEATURE_COLOR, \
    FEATURE_COVARIANCE, FEATURE_ROTATION_MATRIX
from data_utils.utils import quaternion_to_rotation_matrix, rotation_matrix_to_quaternion, \
    farthest_point_sampling, covariance_matrix, strip_symmetric
from plyfile import PlyData, PlyElement
from scipy.special import expit, logit


class GaussianModel:

    @classmethod
    def empty(cls):
        return cls(np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 4)), np.empty((0, 1)), np.empty((0, 3)), np.empty((0, 45)), np.empty(0, dtype=np.int8))

    @classmethod
    def load_from(cls, path, extra_features=None):
        if extra_features is None:
            extra_features = []
        model = GaussianModel.empty()
        model.load_ply(path, extra_features)
        return model

    def __init__(self, xyz, scale, rotation, opacity, features_dc, features_rest, label):
        self._xyz = xyz
        self._scale = scale
        self._rotation = rotation
        self._opacity = opacity
        self._features_dc = features_dc
        self._features_rest = features_rest
        self._label = label

    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_scale(self):
        return self._scale
    
    @property
    def get_rotation_quat(self):
        return self._rotation

    @property
    def get_rotation_matrix(self):
        return quaternion_to_rotation_matrix(self._rotation)

    @property
    def get_covariance(self):
        return strip_symmetric(covariance_matrix(self._scale, self._rotation))
    
    @property
    def get_opacity(self):
        return self._opacity

    @property
    def get_label(self):
        return self._label

    @property
    def get_aabb(self):
        if self._rotation.size > 0 and self._scale.size > 0:
            splat_aabbs = self.__get_splats_aabbs(self._xyz, self._rotation, self._scale)
            object_aabb_min = np.min(splat_aabbs[:, 0, :], axis=0)
            object_aabb_max = np.max(splat_aabbs[:, 1, :], axis=0)
            return np.array([object_aabb_min, object_aabb_max])
        else:
            xyz_min = np.min(self._xyz, axis=0)
            xyz_max = np.max(self._xyz, axis=0)
            return np.array([xyz_min, xyz_max])

    @staticmethod
    def __get_splats_aabbs(xyz, rotations, scales):
        # Generating quaternion matrix
        w, x, y, z = rotations.T
        rotation_matrices = np.array([
            [1 - 2 * y ** 2 - 2 * z ** 2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x ** 2 - 2 * z ** 2, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x ** 2 - 2 * y ** 2]
        ]).transpose(2, 1, 0)

        cube_vertices = np.array(
            [[[-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1], [-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1]]])
        vertices = xyz[:, np.newaxis, :] + np.matmul(cube_vertices * scales[:, np.newaxis, :], rotation_matrices)

        aabb_mins = np.min(vertices, axis=1)
        aabb_maxs = np.max(vertices, axis=1)

        return np.stack([aabb_mins, aabb_maxs], axis=1)

    def __len__(self):
        return len(self._xyz)

    def __add__(self, other):
        if not isinstance(other, GaussianModel):
            raise ValueError("Both objects must be an instance of GaussianModel")

        new_xyz = np.concatenate((self._xyz, other._xyz), axis=0)
        new_scale = np.concatenate((self._scale, other._scale), axis=0)
        new_rotation = np.concatenate((self._rotation, other._rotation), axis=0)
        new_opacity = np.concatenate((self._opacity, other._opacity), axis=0)
        new_features_dc = np.concatenate((self._features_dc, other._features_dc), axis=0)
        new_features_rest = np.concatenate((self._features_rest, other._features_rest), axis=0)
        new_label = np.concatenate((self._label, other._label), axis=0)

        return GaussianModel(new_xyz, new_scale, new_rotation, new_opacity, new_features_dc, new_features_rest, new_label)

    def translated(self, translation):
        new_xyz = self._xyz + translation[np.newaxis, :]
        return GaussianModel(new_xyz, self._scale, self._rotation, self._opacity, self._features_dc, self._features_rest, self._label)

    def rotated(self, R):
        """
        Rotates the model.

        Parameters:
        R : np.ndarray
            A [3, 3] rotation matrix
        """

        new_xyz = np.dot(self._xyz, R.T)

        if self._rotation.size > 0:
            rotation_matrix = quaternion_to_rotation_matrix(self._rotation)
            new_rotation_matrix = R @ rotation_matrix
            new_rotation = rotation_matrix_to_quaternion(new_rotation_matrix)
        else:
            new_rotation = self._rotation

        return GaussianModel(new_xyz, self._scale, new_rotation, self._opacity, self._features_dc, self._features_rest, self._label)

    def scaled(self, scale_vector):
        """
        Scales the model.

        Parameters:
        scale_vector : np.ndarray
            A [3] array representing the scale factors along each axis.
        """
        new_xyz = self._xyz * scale_vector[np.newaxis, :]
        new_scale = self._scale * scale_vector[np.newaxis, :] if self._scale.size > 0 else self._scale
        return GaussianModel(new_xyz, new_scale, self._rotation, self._opacity, self._features_dc, self._features_rest, self._label)

    def shuffled(self):
        indices = np.arange(len(self))
        np.random.shuffle(indices)
        return self.resampled(indices)

    def uniformly_sampled(self, samples):
        selected_ids = np.random.choice(np.arange(len(self)), size=samples, replace=False)
        return self.resampled(selected_ids)

    def fps_sampled(self, samples):
        sampled_ids = farthest_point_sampling(self._xyz, samples)
        return self.resampled(sampled_ids)

    def resampled(self, indices):
        """
        Resamples all attribute arrays of the model according to the given indices.

        Parameters:
        indices : np.ndarray
            An array of indices indicating the new order of the elements.
        """
        new_xyz = self._xyz[indices]
        new_scale = self._scale[indices] if self._scale.size > 0 else self._scale
        new_rotation = self._rotation[indices] if self._rotation.size > 0 else self._rotation
        new_opacity = self._opacity[indices] if self._opacity.size > 0 else self._opacity
        new_features_dc = self._features_dc[indices] if self._features_dc.size > 0 else self._features_dc
        new_features_rest = self._features_rest[indices] if self._features_rest.size > 0 else self._features_rest
        new_label = self._label[indices] if self._label.size > 0 else self._label

        return GaussianModel(new_xyz, new_scale, new_rotation, new_opacity, new_features_dc, new_features_rest, new_label)

    def normalized(self):
        return GaussianModel(self._xyz, np.exp(self._scale), self._rotation / np.linalg.norm(self._rotation, axis=1)[:, np.newaxis], expit(self._opacity), self._features_dc, self._features_rest, self._label)

    def denormalized(self):
        return GaussianModel(self._xyz, np.log(self._scale), self._rotation, logit(self._opacity), self._features_dc, self._features_rest, self._label)

    def with_xyz(self, xyz):
        return GaussianModel(xyz, self._scale, self._rotation, self._opacity, self._features_dc, self._features_rest, self._label)

    def with_label(self, label):
        new_labels = np.repeat(label, len(self))
        return self.with_labels(new_labels)

    def with_labels(self, labels):
        return GaussianModel(self._xyz, self._scale, self._rotation, self._opacity, self._features_dc, self._features_rest, labels)

    def with_color(self, color):
        new_features_dc = np.tile(color, (len(self), 1))
        return self.with_features_dc(new_features_dc)

    def with_features_dc(self, features_dc):
        return GaussianModel(self._xyz, self._scale, self._rotation, self._opacity, features_dc,
                             self._features_rest, self._label)

    def with_features_rest(self, features_rest):
        return GaussianModel(self._xyz, self._scale, self._rotation, self._opacity, self._features_dc,
                             features_rest, self._label)

    def with_features_rest_zeroed(self):
        return self.with_features_rest(np.zeros(self._features_rest.shape))

    def with_color_from_label(self, label_to_color):
        color_map = np.array([label_to_color[label] for label in sorted(label_to_color)])
        color_map = color_map.reshape(-1, 3)
        new_features_dc = color_map[self._label]
        return self.with_features_dc(new_features_dc)

    def load_ply(self, path, extra_features):
        plydata = PlyData.read(path)

        self._xyz = np.stack((
            np.asarray(plydata.elements[0]["x"]),
            np.asarray(plydata.elements[0]["y"]),
            np.asarray(plydata.elements[0]["z"])
        ), axis=1)

        if FEATURE_SCALE in extra_features or FEATURE_COVARIANCE in extra_features:
            scales = np.zeros((len(self), 3))
            for i in range(3):
                scales[:, i] = np.array(plydata.elements[0][f"scale_{i}"])
            self._scale = scales

        if FEATURE_ROTATION_QUAT in extra_features or FEATURE_ROTATION_MATRIX in extra_features or FEATURE_COVARIANCE in extra_features:
            rotations = np.zeros((len(self), 4))
            for i in range(4):
                rotations[:, i] = np.array(plydata.elements[0][f"rot_{i}"])
            self._rotation = rotations

        if FEATURE_OPACITY in extra_features:
            self._opacity = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        if FEATURE_COLOR in extra_features:
            features_dc = np.zeros((len(self), 3))
            for i in range(3):
                features_dc[:, i] = np.array(plydata.elements[0][f"f_dc_{i}"])
            self._features_dc = features_dc

        if FEATURE_REST in extra_features:
            features_rest = np.zeros((len(self), 45))
            for i in range(45):
                features_rest[:, i] = np.array(plydata.elements[0][f"f_rest_{i}"])
            self._features_rest = features_rest

    def save_ply(self, path, include_normals=False):
        attr_names = ['x', 'y', 'z']
        attr_values = [self._xyz]
        if include_normals:
            attr_names.extend(['nx', 'ny', 'nz'])
            attr_values.append(np.zeros_like(self._xyz))
        if self._features_dc.size > 0:
            for i in range(3):
                attr_names.append('f_dc_{}'.format(i))
            attr_values.append(self._features_dc)
        if self._features_rest.size > 0:
            for i in range(45):
                attr_names.append('f_rest_{}'.format(i))
            attr_values.append(self._features_rest)
        if self._opacity.size > 0:
            attr_names.append('opacity')
            attr_values.append(self._opacity)
        if self._scale.size > 0:
            for i in range(3):
                attr_names.append('scale_{}'.format(i))
            attr_values.append(self._scale)
        if self._rotation.size > 0:
            for i in range(4):
                attr_names.append('rot_{}'.format(i))
            attr_values.append(self._rotation)

        dtype = [(attr_name, 'f4') for attr_name in attr_names]
        attributes = np.concatenate(tuple(attr_values), axis=1)
        elements = np.empty(len(self), dtype=dtype)
        elements[:] = [tuple(attr) for attr in attributes]
        ply_element = PlyElement.describe(elements, 'vertex')
        PlyData([ply_element]).write(path)
