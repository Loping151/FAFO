import bpy
import sys

argv = sys.argv
argv = argv[argv.index("--") + 1:] 
glb_path = argv[0]
output_path = './tmp/render.png'

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

bpy.data.scenes["Scene"].render.image_settings.file_format = 'PNG'

bpy.ops.import_scene.gltf(filepath=glb_path)

bpy.context.scene.display_settings.display_device = 'sRGB'

bpy.context.scene.render.engine = 'BLENDER_EEVEE'

cam = bpy.data.cameras.new("Camera")
cam_obj = bpy.data.objects.new("Camera", cam)
bpy.context.scene.collection.objects.link(cam_obj)
bpy.context.scene.camera = cam_obj

light_data = bpy.data.lights.new(name="Light", type='POINT')
light_object = bpy.data.objects.new(name="Light", object_data=light_data)
bpy.context.collection.objects.link(light_object)
light_object.location = (0, -3, 5)

bpy.data.scenes["Scene"].render.filepath = output_path

imported_obj = bpy.context.selected_objects[0]

obj_center = imported_obj.location

cam_obj.location = (obj_center.x, obj_center.y - 3, obj_center.z + 1)

def look_at(obj_camera, target_location):
    direction = target_location - obj_camera.location
    obj_camera.rotation_euler[0] = 1.5708
    obj_camera.rotation_euler[1] = 0
    obj_camera.rotation_euler[2] = 1.5708

look_at(cam_obj, obj_center)

bpy.ops.render.render(write_still=True)
