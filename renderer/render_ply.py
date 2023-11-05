import open3d as o3d
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def render_ply(ply_path):
    pcd = o3d.io.read_point_cloud(ply_path)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False) 
    vis.add_geometry(pcd)

    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    image = vis.capture_screen_float_buffer(do_render=True)
    vis.destroy_window()

    plt_image = np.asarray(image)
    plt_image = (plt_image * 255).astype(np.uint8)
    plt_image = Image.fromarray(plt_image)

    plt_image.save(ply_path[:-4]+'.png')
    
    return plt_image, ply_path[:-4]+'.png'

