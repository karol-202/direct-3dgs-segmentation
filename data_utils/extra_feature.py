FEATURE_ROTATION_QUAT = 'rotation_quat'
FEATURE_ROTATION_MATRIX = 'rotation_matrix'
FEATURE_SCALE = 'scale'
FEATURE_COVARIANCE = 'covariance'
FEATURE_OPACITY = 'opacity'
FEATURE_COLOR = 'color'
FEATURE_REST = 'rest'


class ExtraFeature:
    def __init__(self, name, channels, getter):
        self.name = name
        self.channels = channels
        self.getter = getter

    @staticmethod
    def rotation_quat():
        return ExtraFeature(FEATURE_ROTATION_QUAT, 4, lambda m: m.get_rotation_quat)

    @staticmethod
    def rotation_matrix():
        return ExtraFeature(FEATURE_ROTATION_MATRIX, 9, lambda m: m.get_rotation_matrix.reshape(-1, 9))

    @staticmethod
    def scale():
        return ExtraFeature(FEATURE_SCALE, 3, lambda m: m.get_scale)

    @staticmethod
    def covariance():
        return ExtraFeature(FEATURE_COVARIANCE, 6, lambda m: m.get_covariance)

    @staticmethod
    def opacity():
        return ExtraFeature(FEATURE_OPACITY, 1, lambda m: m.get_opacity)

    @staticmethod
    def feature_by_name(name):
        if name == FEATURE_ROTATION_QUAT:
            return ExtraFeature.rotation_quat()
        elif name == FEATURE_ROTATION_MATRIX:
            return ExtraFeature.rotation_matrix()
        elif name == FEATURE_SCALE:
            return ExtraFeature.scale()
        elif name == FEATURE_COVARIANCE:
            return ExtraFeature.covariance()
        elif name == FEATURE_OPACITY:
            return ExtraFeature.opacity()
        else:
            raise ValueError(f'Unknown feature name: {name}')
