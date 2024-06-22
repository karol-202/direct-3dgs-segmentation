from datasets.base_mesh_dataset import BaseMeshDataset
from data_utils.utils import split_into_random_subsets
from data_utils.scene_composer import compose_scene


class ComposedMeshDataset(BaseMeshDataset):
    def __init__(self, model_paths, class2label, num_point=4096):
        super().__init__(model_paths, class2label, num_point)

        self.scenes = split_into_random_subsets(self.models, min_subset_size=3, max_subset_size=5)

    def __getitem__(self, idx):
        scene = self.scenes[idx]

        samples_per_model = [int(self.num_point / len(scene)) for model in scene]
        # Make sure the samples sum up to num_point
        samples_per_model[-1] = self.num_point - sum(samples_per_model[:-1])

        scene_models = [model.sample_from_faces(samples_per_model[i]) for i, model in enumerate(self.scenes[idx])]
        composed_scene = compose_scene(scene_models)
        selected_points, selected_labels, _ = self.get_data_for_model_with_uniform_sampling(composed_scene, [])
        return selected_points, selected_labels

    def __len__(self):
        return len(self.scenes)
