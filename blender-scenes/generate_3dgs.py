import os.path
import subprocess

SCENES_PATH = '/home/karol/dev/tud/Y3Q4_RP/direct-3dgs/blender-scenes/scenes/single-1'
KEEP_OLD_GS = True
GS_TRAIN_PATH = '../../gaussian-splatting/train.py'
ITERS = 10_000
RENDERS_DIRECTORY = 'renders'
GS_DIRECTORY = '3dgs'
MIN_MODEL_NUMBER = 0
MAX_MODEL_NUMBER = 1000

directories = sorted(os.scandir(SCENES_PATH), key=lambda dir: dir.name)

for directory in directories:
    if KEEP_OLD_GS and GS_DIRECTORY in [child.name for child in os.scandir(directory.path) if child.is_dir()]:
        continue

    directory_number = int(directory.name.split('_')[-1].lstrip("0"))
    if directory_number < MIN_MODEL_NUMBER or directory_number > MAX_MODEL_NUMBER:
        continue

    renders_path = os.path.join(directory.path, 'renders')
    if not os.path.isdir(renders_path):
        raise FileNotFoundError(f"'renders' directory does not exist in {directory.path}")

    output_path = os.path.join(directory.path, GS_DIRECTORY)
    cmd = (f'source /etc/profile.d/conda.sh'
           f' && conda activate gaussian_splatting'
           f' && python {GS_TRAIN_PATH} -s {renders_path} --iterations {ITERS} -m {output_path}')
    subprocess.run(cmd, shell=True)
