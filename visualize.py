import os
import random

import numpy as np
import torch

from data_utils.extra_feature import ExtraFeature, FEATURE_ROTATION_QUAT, FEATURE_OPACITY, FEATURE_SCALE
from data_utils.gaussian_model import GaussianModel
from data_utils.utils import dotdict, read_split_file
from data_utils.scene_composer import compose_scene
from train_3dgs import NUM_CLASSES, create_environment, CLASSES, CLASS2LABEL

DATA_PATH = '/3dgs-dataset'
MODEL = 'pointnet2_sem_seg'
NUM_POINT = 4096
EXTRA_FEATURES_TO_TRAIN = [FEATURE_OPACITY, FEATURE_SCALE, FEATURE_ROTATION_QUAT]
EXTRA_FEATURES_TO_LOAD = [*EXTRA_FEATURES_TO_TRAIN, 'color', 'rest']

CLASSES2COLORS = {
    'bathtub': [0, 0, 0],
    'bed': [0, 1, 0],
    'chair': [0, 0, 1],
    'desk': [0, 0, 0],
    'dresser': [0, 0, 0],
    'monitor': [0, 0, 0],
    'night_stand': [0, 0, 0],
    'sofa': [0, 0, 0],
    'table': [0, 0, 0],
    'toilet': [1, 0, 0],
}
LABELS2COLORS = {CLASS2LABEL[cls]: CLASSES2COLORS[cls] for cls in CLASSES}

random.seed(43)
np.random.seed(43)

models = [
    GaussianModel.load_from('/3dgs-dataset/bed/bed_0001/point_cloud/iteration_15000/point_cloud.ply', EXTRA_FEATURES_TO_LOAD).normalized().with_label(CLASS2LABEL['bed']),
    GaussianModel.load_from('/3dgs-dataset/toilet/toilet_0001/point_cloud/iteration_15000/point_cloud.ply', EXTRA_FEATURES_TO_LOAD).normalized().with_label(CLASS2LABEL['toilet']),
    GaussianModel.load_from('/3dgs-dataset/chair/chair_0003/point_cloud/iteration_15000/point_cloud.ply', EXTRA_FEATURES_TO_LOAD).normalized().with_label(CLASS2LABEL['chair']),
]
composed_model = compose_scene(models) # .with_label(-1)

args = dotdict({
    'model': MODEL,
    'dataset_type': '3DGS',
    'log_dir': 'visualization_3dgs_uniform_pos+op+scale+rot_1',
    'batch_size': 8,
    'learning_rate': 0,
    'gpu': '0',
    'optimizer': 'Adam',
    'weight_decay_rate': 0,
    'npoint': NUM_POINT,
    'eval_after_epoch': False,
    'extra_features': EXTRA_FEATURES_TO_TRAIN
})
train_scene_paths = read_split_file(DATA_PATH, 'train.txt')[:10]
test_scene_paths = read_split_file(DATA_PATH, 'test.txt')[:10]
env = create_environment(args, train_scene_paths, test_scene_paths)

points, _, resampled_model = env.test_dataset.get_data_for_model_with_uniform_sampling(composed_model, [ExtraFeature.feature_by_name(feature_name) for feature_name in EXTRA_FEATURES_TO_TRAIN])
points = torch.Tensor(points)
points = points.float().cuda()
points = points.T.reshape(1, points.shape[1], points.shape[0])  # [N, C] -> [1, C, N]

seg_pred, _ = env.classifier(points)
pred_val = seg_pred.contiguous().cpu().data.numpy()
seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)
pred_val = np.argmax(pred_val, 2)[0, :]

colored_model = composed_model.with_color_from_label(LABELS2COLORS).scaled(np.repeat(5, 3)).denormalized()
colored_model.save_ply('/home/karol/rp/direct-3dgs/blender-scenes/scenes/visualization/point_cloud/iteration_15000/point_cloud.ply', include_normals=True)
