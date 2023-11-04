import numpy as np
from stl import mesh
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as mplot3d
from PIL import Image


def render_stl(path):
    your_mesh = mesh.Mesh.from_file(path)

    min_x, max_x = np.min(your_mesh.x), np.max(your_mesh.x)
    min_y, max_y = np.min(your_mesh.y), np.max(your_mesh.y)
    min_z, max_z = np.min(your_mesh.z), np.max(your_mesh.z)

    width = max_x - min_x
    height = max_y - min_y

    output_width = 800
    output_height = int(output_width * (height / width))

    fig = plt.figure(figsize=(output_width / 100, output_height / 100))
    fig.set_dpi(100)
    ax = fig.add_subplot(111, projection='3d')

    ax.view_init(elev=20, azim=30)

    ax.add_collection3d(mplot3d.art3d.Poly3DCollection(your_mesh.vectors, facecolors='gray'))

    margin = 10
    ax.set_xlim([min_x - margin, max_x + margin])
    ax.set_ylim([min_y - margin, max_y + margin])
    ax.set_zlim([min_z - margin, max_z + margin])

    ax.axis('off')

    ax.set_facecolor((1, 1, 1, 0))
    fig.patch.set_facecolor((1, 1, 1, 0))

    fig.canvas.draw()
    img_arr = np.array(fig.canvas.renderer.buffer_rgba())
    img = Image.fromarray(img_arr)

    img.save(path[:-4]+'.png')

    return img, path[:-4]+'.png'