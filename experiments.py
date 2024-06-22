import hashlib
import os
import random

import numpy as np

from data_utils.extra_feature import FEATURE_COVARIANCE, FEATURE_OPACITY, FEATURE_SCALE, FEATURE_ROTATION_QUAT, \
    FEATURE_ROTATION_MATRIX
from data_utils.utils import read_split_file, dotdict, load_json, save_json
from train_3dgs import create_environment, train, evaluate, close_environment

base_args = dotdict({
    'model': 'pointnet2_sem_seg',
    'batch_size': 8,
    'epoch': 200,
    'gpu': '0',
    'optimizer': 'Adam',
    'weight_decay_rate': 0.0001,
    'npoint': 4096,
    'lr_step_size': 10,
    'eval_after_epoch': False,
})


def experiments_additional_args(run):
    return [
        dotdict({
            'data_path': '../raw-datasets/ModelNet10/classes-off',
            'dataset_type': 'SampledMesh',
            'log_dir': f'exp_pcd_{run + 1}',
            'learning_rate': 0.003,
            'lr_decay': 0.8,
        }),
        dotdict({
            'data_path': '/3dgs-dataset',
            'dataset_type': '3DGS',
            'log_dir': f'exp_3dgs_uniform_pos_{run+1}',
            'extra_features': None,
            'sampling': 'uniform',
            'learning_rate': 0.003,
            'lr_decay': 0.9,
        }),
        dotdict({
            'data_path': '/3dgs-dataset',
            'dataset_type': '3DGS',
            'log_dir': f'exp_3dgs_uniform_pos+op_{run+1}',
            'extra_features': [FEATURE_OPACITY],
            'sampling': 'uniform',
            'learning_rate': 0.003,
            'lr_decay': 0.9,
        }),
        dotdict({
            'data_path': '/3dgs-dataset',
            'dataset_type': '3DGS',
            'log_dir': f'exp_3dgs_uniform_pos+op+scale+rot_{run+1}',
            'extra_features': [FEATURE_OPACITY, FEATURE_SCALE, FEATURE_ROTATION_QUAT],
            'sampling': 'uniform',
            'learning_rate': 0.003,
            'lr_decay': 0.9,
        }),
        dotdict({
            'data_path': '/3dgs-dataset',
            'dataset_type': '3DGS',
            'log_dir': f'exp_3dgs_uniform_pos+op+cov_{run + 1}',
            'extra_features': [FEATURE_OPACITY, FEATURE_COVARIANCE],
            'sampling': 'uniform',
            'learning_rate': 0.003,
            'lr_decay': 0.9,
        }),
        dotdict({
            'data_path': '/3dgs-dataset',
            'dataset_type': '3DGS',
            'log_dir': f'exp_3dgs_uniform_pos+op+scale+rotmat_{run + 1}',
            'extra_features': [FEATURE_OPACITY, FEATURE_SCALE, FEATURE_ROTATION_MATRIX],
            'sampling': 'uniform',
            'learning_rate': 0.003,
            'lr_decay': 0.9,
        }),
    ]


runs_per_experiment = 3
selected_runs = []
experiments_args = [base_args + experiment for run in range(runs_per_experiment) for experiment in experiments_additional_args(run) if experiment.log_dir in selected_runs]

exp_set_name = 'exp'
directory_name = f'log/sem_seg/{exp_set_name}'
os.makedirs(directory_name, exist_ok=True)
progress_filename = f'{directory_name}/progress.json'
progress = load_json(progress_filename) or {}

for experiment_args in experiments_args:
    experiment_name = experiment_args.log_dir
    if experiment_name in progress:
        continue

    data_path = experiment_args.data_path
    train_scene_paths = read_split_file(data_path, 'train.txt')
    test_scene_paths = read_split_file(data_path, 'test.txt')

    seed = int(hashlib.md5(experiment_name.encode('utf-8')).hexdigest(), 16) % (2**32)
    random.seed(seed)
    np.random.seed(seed)

    env = create_environment(experiment_args, train_scene_paths, test_scene_paths)
    train(env, experiment_args)
    results = evaluate(env, experiment_args)
    close_environment(env)

    progress[experiment_name] = {
        'accuracy': results.accuracy,
        'mIoU': results.mIoU,
    }
    save_json(progress_filename, progress)
