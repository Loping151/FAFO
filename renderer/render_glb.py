import os
from PIL import Image
import subprocess
import pyrender
import trimesh
import numpy as np
import os

def render_glb(path):
    if not os.path.exists('./tmp'):
        os.mkdir('./tmp')
    # subprocess.call(['blender', '-b', '-P', './renderer/render_glb_blender.py', '--', path])
    # img = Image.open('./tmp/render.png')
    img = render(path)
    img.save(path[:-4]+'.png')
    return img, path[:-4]+'.png'


def render(path):
    trimesh_scene = trimesh.load(path)

    scene = pyrender.Scene.from_trimesh_scene(trimesh_scene)

    renderer = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)

    camera_pose = np.array([
        [0.0,  0.0, -1.0,  10.0],
        [1.0,  0.0,  0.0,  2.0],
        [0.0, -1.0,  0.0,  2.0],
        [0.0,  0.0,  0.0,  1.0],
    ])

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    scene.add(camera, pose=camera_pose)

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=camera_pose)

    color, _ = renderer.render(scene)

    renderer.delete()
    
    return Image.fromarray(color)
