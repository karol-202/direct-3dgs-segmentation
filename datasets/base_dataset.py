import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm


class BaseDataset(Dataset):
    def __init__(self, model_paths, class2label, num_point=4096):
        super().__init__()
        self.num_point = num_point

        self.models = []
        label_weights = np.zeros(len(class2label))

        for model_path in tqdm(model_paths, total=len(model_paths)):
            scene_path_parts = model_path.split('/')
            class_name, dir_name = scene_path_parts[-2], scene_path_parts[-1]

            label = class2label[class_name]

            model = self.load_model(model_path, label)
            self.models.append(model)

            label_weights[label] += len(model)

        label_weights = label_weights.astype(np.float32)
        label_weights = label_weights / np.sum(label_weights)
        self.label_weights = np.power(np.amax(label_weights) / label_weights, 1 / 3.0)
        print(self.label_weights)

        print("Totally {} models.".format(len(self.models)))

    def load_model(self, model_path, label):
        raise NotImplementedError()

    def get_data_for_model_with_uniform_sampling(self, model, extra_features):
        resampled_model = model.uniformly_sampled(self.num_point)
        return self.get_data_for_model(resampled_model, extra_features)

    def get_data_for_model(self, model, extra_features):
        assert len(model) == self.num_point

        selected_points = np.zeros((self.num_point, self.get_channels_count_for_features(extra_features)))
        selected_points[:, 0:3] = model.get_xyz
        channel_index = 3
        for feature in extra_features:
            num_channels, channels_getter = feature.channels, feature.getter
            selected_points[:, channel_index:channel_index + num_channels] = channels_getter(model)
            channel_index += num_channels

        selected_labels = model.get_label

        return selected_points, selected_labels, model

    def get_channels_count_for_features(self, extra_features):
        channels = 3
        for feature in extra_features:
            channels += feature.channels
        return channels
