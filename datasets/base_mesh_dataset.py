from datasets.base_dataset import BaseDataset
from data_utils.mesh import Mesh

MESH_EXTENSION = '.off'


class BaseMeshDataset(BaseDataset):
    def __init__(self, model_paths, class2label, num_point=4096):
        super().__init__(model_paths, class2label, num_point)

    def load_model(self, model_path, label):
        mesh_path = model_path + MESH_EXTENSION
        return Mesh(mesh_path, label)

    @property
    def get_channels_count(self):
        return 3
