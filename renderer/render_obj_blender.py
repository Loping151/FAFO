import bpy
import sys

argv = sys.argv
argv = argv[argv.index("--") + 1:]

obj_path = argv[0]
output_path = './tmp/render.png'

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

bpy.ops.import_scene.obj(filepath=obj_path)

bpy.context.scene.render.engine = 'CYCLES'

cam = bpy.data.cameras.new("Camera")
cam_obj = bpy.data.objects.new("Camera", cam)
bpy.context.scene.collection.objects.link(cam_obj)
bpy.context.scene.camera = cam_obj
cam_obj.location = (0, -3, 1)
cam_obj.rotation_euler = (1.10871, 0, 0.785398)

light_data = bpy.data.lights.new(name="Light", type='POINT')
light_object = bpy.data.objects.new(name="Light", object_data=light_data)
bpy.context.collection.objects.link(light_object)
light_object.location = (0, -3, 5)

bpy.context.scene.render.resolution_x = 1920
bpy.context.scene.render.resolution_y = 1080

bpy.context.scene.render.filepath = output_path

bpy.ops.render.render(write_still=True)
