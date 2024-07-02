# Semantic 3D segmentation of 3D Gaussian Splats 

This is the code repository for my Bachelor thesis: _Semantic 3D segmentation of 3D Gaussian Splats: Assessing existing point cloud segmentation techniques on semantic segmentation of synthetic 3D Gaussian Splats scenes_.

## Paper abstract
_[3D Gaussian Splatting (3DGS)](https://github.com/graphdeco-inria/gaussian-splatting) is a promising 3D reconstruction
and novel-view synthesis technique. However, the field of semantic 3D segmentation of 3D Gaussian Splats scenes remains
largely unexplored. This paper discusses the challenges of performing 3D segmentation directly on 3D Gaussian Splats,
introduces a new dataset facilitating evaluation of 3DGS semantic segmentation and proposes use of PointNet++,
initially developed for point cloud segmentation, as a 3DGS segmentation model. As the results show, PointNet++
is also capable of performing 3DGS segmentation with performance close to the performance achieved in point cloud
segmentation tasks. When taking into account only the positions, 3D Gaussian splats appear to be more difficult
for PointNet++ to process than point clouds sampled from mesh faces, possibly due to their irregularity.
However, as shown in the paper, inclusion of size, rotation and opacity of each splat allows PointNet++ to achieve
nearly 87% of accuracy, outperforming PointNet++ on point clouds sampled from meshes._

Link to the full paper: https://repository.tudelft.nl/record/uuid:387de885-1c53-4126-af25-19ebaa530a04

## Overview
This repository contains:
- scripts for creating the 3DGS dataset used in the paper: `blender-scenes/`
- PyTorch PointNet++ implementation from https://github.com/yanx27/Pointnet_Pointnet2_pytorch: `models/`
- 3DGS dataset loaders: `datasets/`
- script for training PointNet++ on 3DGS data: `train_3dgs.py`
- script for performing hyperparameter tuning: `tune_hyperparams.py`
- script for running multiple experiments: `experiments.py`
- script for visualizing 3DGS segmentation performed by the model: `visualize.py`

## Datasets
The 3DGS dataset used in the paper can be downloaded from: https://doi.org/10.4121/3eabd3f5-d814-48be-bbff-b440f2d48a2b.

The directory denoted by `<DATASET_PATH>` in the rest of this file is expected to have the following structure:
- `<DATASET_PATH>`
  - `bed`
    - `bed_0001`
      - `point_cloud`
        - `iteration_15000`
          - `point_cloud.ply`
    - _rest of the models_
  - _rest of the categories_
  - `train.txt` - list of models belonging to the _train_ split of the dataset
  - `test.txt` - list of models belonging to the _test_ split of the dataset

Additionally, to train the model on point clouds sampled from ModelNet10 meshes (used as the baseline in the paper),
the [ModelNet10 dataset](https://modelnet.cs.princeton.edu/#) is needed.
The directory denoted by `<MODELNET_PATH>` in the rest of this file is expected to have the following structure:
- `<DATASET_PATH>`
  - `bed`
    - `bed_0001.off`
    - _rest of the models_
  - _rest of the categories_
  - `train.txt` - list of models belonging to the _train_ split of the dataset
  - `test.txt` - list of models belonging to the _test_ split of the dataset

The `train.txt` and `test.txt` files used in the paper can be found in `data/`.
They are based on the original train/test split from the [ModelNet10 dataset](https://modelnet.cs.princeton.edu/#).

## How to use

### Prerequisites
Running most of the code in this repository (including dataset generation and model training) requires a graphics card
with CUDA support.

### Dataset generation
The 3DGS dataset linked above can be also recreated by using scripts from this repository.

Beware! This can take a lot of time.

For dataset generation:
- Blender should be installed (tested with 4.1 version)
- the [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) repository should be cloned
  and set up according to its README

To generate the dataset:
- run `blender-scenes/create-single-scenes.py` to create Blender scenes for each ModelNet10 object
- run `blender-scenes/render-all.py` to render all Blender scenes into images
- run `blender-scenes/generate_3dgs.py` to create 3DGS scenes for each of the models

The paths in the scripts above need to be adjusted to match the local setup and directory structure.

### Training
To use the code from this repository, a Conda environment must be set up.
Assuming Conda is installed, the following command can be run to create the environment:
```bash
conda env create --file environment.yml
```

When the environment is created, it should be activated:
```bash
conda activate direct-3dgs
```

Training a single PointNet++ model can be performed using the `train_3dgs.py` script.

Training on 3DGS data using position, opacity, scale and rotation (quaternion):
```bash
python train_3dgs.py <DATASET_PATH> --model pointnet2_sem_seg --log_dir <EXPERIMENT_NAME> --batch_size <BATCH_SIZE> --epoch <EPOCHS> --extra_features opacity,scale,rotation_quat
```

Training on point clouds sampled from ModelNet10 meshes (the baseline):
```bash
python train_3dgs.py <MODELNET_PATH> --model pointnet2_sem_seg --log_dir <EXPERIMENT_NAME> --batch_size <BATCH_SIZE> --epoch <EPOCHS> --dataset_type SampledMesh
```

The trained model and training logs will be saved in `log/sem_seg/<EXPERIMENT_NAME>`.
