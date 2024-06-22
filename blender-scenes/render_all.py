import subprocess
import os

SCENES_PATH = '/home/karol/dev/tud/Y3Q4_RP/direct-3dgs/blender-scenes/scenes/single-1'
KEEP_OLD_RENDERS = True
RENDER_SCRIPT = 'b-360_render-single.py'
MIN_MODEL_NUMBER = 0
MAX_MODEL_NUMBER = 1000

directories = sorted(os.scandir(SCENES_PATH), key=lambda dir: dir.name)

for directory in directories:
    if KEEP_OLD_RENDERS and 'renders' in [child.name for child in os.scandir(directory.path) if child.is_dir()]:
        continue

    directory_number = int(directory.name.split('_')[-1].lstrip("0"))
    if directory_number < MIN_MODEL_NUMBER or directory_number > MAX_MODEL_NUMBER:
        continue

    scene_path = os.path.join(directory, 'scene.blend')
    cmd = f'blender -b {scene_path} -P {RENDER_SCRIPT}'
    subprocess.run(cmd, shell=True)
