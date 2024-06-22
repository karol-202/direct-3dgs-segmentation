import os

from datasets.base_dataset import BaseDataset
from data_utils.gaussian_model import GaussianModel

PCD_PATH = 'point_cloud/iteration_15000/point_cloud.ply'


class Base3DGSDataset(BaseDataset):
    def __init__(self, model_paths, class2label, sampling, num_point=4096, extra_features=None):
        self.sampling = sampling
        self.extra_features = extra_features or []
        super().__init__(model_paths, class2label, num_point)

    def load_model(self, model_path, label):
        ply_path = os.path.join(model_path, PCD_PATH)
        extra_feature_names = [feature.name for feature in self.extra_features]
        model = (GaussianModel.load_from(ply_path, extra_feature_names)
                 .normalized()
                 .with_label(label))
        return model.fps_sampled(self.num_point) if self.sampling == 'fps' else model

    @property
    def get_channels_count(self):
        return self.get_channels_count_for_features(self.extra_features)
