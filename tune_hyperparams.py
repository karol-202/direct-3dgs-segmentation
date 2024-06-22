import os

from sklearn.model_selection import KFold, ParameterGrid
import numpy as np

from data_utils.utils import dotdict, read_split_file, load_json, save_json
from train_3dgs import train, create_environment, evaluate, close_environment

data_path = '../raw-datasets/ModelNet10/classes-off'  # '/3dgs-dataset'
scene_paths = np.array(read_split_file(data_path, 'train.txt'))

run_name = 'hyper_pcd'
directory_name = f'log/sem_seg/{run_name}'
os.makedirs(directory_name, exist_ok=True)

folds_selector = [0]
param_id_selector = [pid for pid in range(1000)]

progress_filename = f'{directory_name}/progress.json'
results_filename = f'{directory_name}/results.json'
param_grid = {
    'lr_decay_rate': [0.9, 0.8, 0.7],
    'lr_initial': [0.001, 0.003, 0.01],
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
param_grid = ParameterGrid(param_grid)

progress = load_json(progress_filename) or {}
results = []

for param_idx, params in enumerate(param_grid):
    if param_idx not in param_id_selector:
        continue

    param_key = str(param_idx)
    if param_key not in progress:
        progress[param_key] = {'params': params, 'fold_results': [], 'completed_folds': 0}

    fold_results = progress[param_key]['fold_results']
    completed_folds = progress[param_key]['completed_folds']

    for fold_idx, (train_index, val_index) in enumerate(kf.split(scene_paths)):
        if fold_idx < completed_folds:
            continue
        if fold_idx not in folds_selector:
            continue

        scenes_train, scenes_val = scene_paths[train_index], scene_paths[val_index]

        args = dotdict({
            'model': 'pointnet2_sem_seg',
            'dataset_type': 'SampledMesh',
            'log_dir': f'{run_name}_params' + str(param_idx) + '_fold' + str(fold_idx),
            'batch_size': 8,
            'epoch': 200,
            'learning_rate': params['lr_initial'],
            'gpu': '0',
            'optimizer': 'Adam',
            'weight_decay_rate': 0.0001,
            'npoint': 4096,
            'lr_step_size': 10,
            'lr_decay': params['lr_decay_rate'],
            'eval_after_epoch': False
        })
        env = create_environment(args, scenes_train, scenes_val)
        train(env, args)
        eval_results = evaluate(env, args)
        close_environment(env)

        fold_results.append(eval_results.accuracy)
        progress[param_key]['completed_folds'] = fold_idx + 1

        save_json(progress_filename, progress)

    mean_score = np.mean(fold_results)
    results.append((params, mean_score))

save_json(results_filename, results)
