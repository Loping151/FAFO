import os
from PIL import Image
import subprocess

def render_obj(path):
    if not os.path.exists('./tmp'):
        os.mkdir('./tmp')
    subprocess.call(['blender', '-b', '-P', './renderer/render_obj_blender.py', '--', path])
    img = Image.open('./tmp/render.png')
    img.save(path[:-4]+'.png')
    return img, path[:-4]+'.png'