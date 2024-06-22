import os.path
import bpy

MODEL_PATH = 'model.obj'
CLASS_PATH = 'class.txt'

working_dir = os.path.dirname(bpy.data.filepath)
os.chdir(working_dir)

for obj in bpy.data.collections['Objects'].objects:
    bpy.data.objects.remove(obj, do_unlink=True)

with open(CLASS_PATH, 'r') as file:
    class_name = file.read()

bpy.ops.wm.obj_import(filepath=MODEL_PATH)
imported_object = bpy.context.active_object

bpy.ops.collection.objects_remove_all()  # Remove the active object from all collections
bpy.data.collections['Objects'].objects.link(imported_object)

original_dim = imported_object.dimensions

scale_factor = 2 / max(original_dim.x, original_dim.y, original_dim.z)
scale_x = scale_factor
scale_y = scale_factor
scale_z = scale_factor

imported_object.scale = (scale_x, scale_y, scale_z)
imported_object.rotation_euler = (0, 0, 0)
imported_object.location = (0, 0, 0)

mat = bpy.data.materials[class_name]
if imported_object.data.materials:
    imported_object.data.materials[0] = mat
else:
    imported_object.data.materials.append(mat)

bpy.context.view_layer.objects.active = imported_object
bpy.ops.object.mode_set(mode='EDIT')

bpy.ops.uv.smart_project()

bpy.ops.object.mode_set(mode='OBJECT')

bpy.ops.wm.save_mainfile()
