from datasets.base_3dgs_dataset import Base3DGSDataset


class SingleModel3DGSDataset(Base3DGSDataset):
    def __init__(self, model_paths, class2label, sampling, num_point=4096, extra_features=None):
        super().__init__(model_paths, class2label, sampling, num_point, extra_features)

    def __getitem__(self, idx):
        model = self.models[idx]
        selected_points, selected_labels, _ = self.get_data_for_model_with_uniform_sampling(model, self.extra_features)
        return selected_points, selected_labels

    def __len__(self):
        return len(self.models)
