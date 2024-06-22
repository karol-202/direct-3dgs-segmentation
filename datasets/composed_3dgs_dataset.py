from datasets.base_3dgs_dataset import Base3DGSDataset
from data_utils.utils import split_into_random_subsets
from data_utils.scene_composer import compose_scene


class Composed3DGSDataset(Base3DGSDataset):
    def __init__(self, model_paths, class2label, sampling, num_point=4096, extra_features=None):
        super().__init__(model_paths, class2label, sampling, num_point, extra_features)

        self.scenes = split_into_random_subsets(self.models, min_subset_size=3, max_subset_size=5)

    def __getitem__(self, idx):
        scene_models = self.scenes[idx]
        composed_scene = compose_scene(scene_models)
        selected_points, selected_labels, _ = self.get_data_for_model_with_uniform_sampling(composed_scene, self.extra_features)
        return selected_points, selected_labels

    def __len__(self):
        return len(self.scenes)
