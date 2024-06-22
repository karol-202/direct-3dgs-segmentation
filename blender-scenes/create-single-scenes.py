import glob
import os
import shutil
import subprocess

MODELNET_PATH = '/home/karol/dev/tud/Y3Q4_RP/raw-datasets/ModelNet10/classes-obj'
OUTPUT_PATH = '/home/karol/dev/tud/Y3Q4_RP/direct-3dgs/blender-scenes/scenes/single-1'
TEMPLATE_PATH = '/home/karol/dev/tud/Y3Q4_RP/direct-3dgs/blender-scenes/template/scene.blend'

filepaths = glob.glob(os.path.join(MODELNET_PATH, '*/*'))
filepaths = sorted(filepaths)
for filepath in filepaths:
    basename_no_ext = os.path.splitext(os.path.basename(filepath))[0]
    new_dir_path = os.path.join(OUTPUT_PATH, basename_no_ext)
    os.makedirs(new_dir_path, exist_ok=True)

    target_scene_path = os.path.join(new_dir_path, 'scene.blend')
    target_model_path = os.path.join(new_dir_path, 'model.obj')
    target_class_path = os.path.join(new_dir_path, 'class.txt')

    shutil.copy2(TEMPLATE_PATH, target_scene_path)
    shutil.copy2(filepath, target_model_path)

    class_name = basename_no_ext.rsplit('_', 1)[0]
    with open(target_class_path, 'w') as f:
        f.write(class_name)

    blender_cmd = f'blender -b {target_scene_path} -P b-scene-init-single.py'
    subprocess.run(blender_cmd, shell=True)
