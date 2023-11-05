import bpy
import sys

argv = sys.argv
argv = argv[argv.index("--") + 1:]

gltf_file = argv[0]
obj_file = argv[1]

bpy.ops.object.select_all(action='DESELECT')
bpy.ops.object.select_by_type(type='MESH')
bpy.ops.object.delete()

bpy.ops.import_scene.gltf(filepath=gltf_file)

bpy.ops.export_scene.obj(filepath=obj_file, use_selection=False)

# blender -b -P tools/gltf2obj.py -- /home/loping151/Downloads/vintage_leather_sofa/scene.gltf test.obj
