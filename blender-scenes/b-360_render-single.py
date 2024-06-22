# Inspired by: NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis, Ben Mildenhall et al.

import os
import json
import bpy
from math import radians

DEBUG = False

RESOLUTION = 800
RESULTS_PATH = 'renders'
DEPTH_SCALE = 1.4
FORMAT = 'PNG'

CAMERA_ANGLES = [
    [90, 0, 0],
    [50, 0, 0+15],
    [50, 0, 30+15],
    [50, 0, 60+15],
    [50, 0, 90+15],
    [50, 0, 120+15],
    [50, 0, 150+15],
    [50, 0, 180+15],
    [50, 0, 210+15],
    [50, 0, 240+15],
    [50, 0, 270+15],
    [50, 0, 300+15],
    [50, 0, 330+15],
    [20, 0, 0],
    [20, 0, 30],
    [20, 0, 60],
    [20, 0, 90],
    [20, 0, 120],
    [20, 0, 150],
    [20, 0, 180],
    [20, 0, 210],
    [20, 0, 240],
    [20, 0, 270],
    [20, 0, 300],
    [20, 0, 330],
    [-30, 0, 0],
    [-30, 0, 180],
    [-90, 0, 0]
]

def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list


results_path_abs = bpy.path.abspath(f"//{RESULTS_PATH}")
if not os.path.exists(results_path_abs):
    os.makedirs(results_path_abs)

# Data to store in JSON file
out_data = {
    'camera_angle_x': bpy.data.objects['Camera'].data.angle_x,
}

# Render Optimizations
bpy.context.scene.render.use_persistent_data = True

# Set up rendering of depth map.
tree = bpy.context.scene.node_tree

bpy.context.scene.render.image_settings.file_format = str(FORMAT)

# Background
bpy.context.scene.render.dither_intensity = 0.0
bpy.context.scene.render.film_transparent = True


objs = [ob for ob in bpy.context.scene.objects if ob.type in ('EMPTY') and 'Empty' in ob.name]
with bpy.context.temp_override(selected_objects=objs):
    bpy.ops.object.delete()


def parent_obj_to_camera(b_camera):
    origin = (0, 0, 0)
    b_empty = bpy.data.objects.new("Empty", None)
    b_empty.location = (0, 0, 0)
    b_camera.parent = b_empty  # setup parenting

    scn = bpy.context.scene
    scn.collection.objects.link(b_empty)
    bpy.context.view_layer.objects.active = b_empty
    # scn.objects.active = b_empty
    return b_empty


scene = bpy.context.scene
scene.render.resolution_x = RESOLUTION
scene.render.resolution_y = RESOLUTION
scene.render.resolution_percentage = 100

cam = scene.objects['Camera']
cam.location = (0, 4.0, 0)
cam_constraint = cam.constraints.new(type='TRACK_TO')
cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
cam_constraint.up_axis = 'UP_Y'
b_empty = parent_obj_to_camera(cam)
cam_constraint.target = b_empty

scene.render.image_settings.file_format = 'PNG'  # set output format to .png

out_data['frames'] = []

for i in range(0, len(CAMERA_ANGLES)):
    image_path = 'image/r_' + str(i) + '.png'

    scene.render.filepath = f'//{RESULTS_PATH}/{image_path}'

    tree.nodes['Semantic Output'].file_slots.clear()

    camera_angle = CAMERA_ANGLES[i]
    b_empty.rotation_euler[0] = radians(camera_angle[0])
    b_empty.rotation_euler[1] = radians(camera_angle[1])
    b_empty.rotation_euler[2] = radians(camera_angle[2])

    if DEBUG:
        break
    else:
        bpy.ops.render.render(write_still=True)  # render still

    frame_data = {
        'file_path': image_path,
        'transform_matrix': listify_matrix(cam.matrix_world)
    }
    out_data['frames'].append(frame_data)

if not DEBUG:
    with open(results_path_abs + '/transforms.json', 'w') as out_file:
        json.dump(out_data, out_file, indent=4)
