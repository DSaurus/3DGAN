import trimesh
import os
import numpy as np
from tqdm import tqdm
from skimage import measure
from utils.mesh_util import save_obj_mesh
import matplotlib.pyplot as plt
import math

def rotationY(angle):
    return [
            [ math.cos(angle), 0, math.sin(angle)],
            [               0, 1,               0],
            [-math.sin(angle), 0, math.cos(angle)],
           ]

mesh_dir = 'dataset/dh/mesh/'
save_dir = 'dataset/dh/vox_rot/'
os.makedirs(save_dir, exist_ok=True)
for mesh_name in tqdm(os.listdir(mesh_dir)):
    # if os.path.exists(os.path.join(save_dir, mesh_name[:-4] + '.npy')):
    #     continue
    try:
        mesh = trimesh.load(os.path.join(mesh_dir, mesh_name))
    except Exception:
        continue

    v = mesh.vertices
    vmin = np.min(v, axis=0)
    vmax = np.max(v, axis=0)
    center = (vmin+vmax)/2

    transform = np.eye(4)
    transform[0, 3] = -center[0]
    transform[1, 3] = -center[1]
    transform[2, 3] = -center[2]
    mesh.apply_transform(transform)

    length = np.max(vmax - vmin)*1.1
    transform = np.eye(4)
    for i in range(3):
        transform[i, i] /= length/2
    mesh.apply_transform(transform)
    
    for j in range(30):
        r = math.acos(-1)* (12 / 180)
        transform = np.eye(4)
        rot = np.array(rotationY(r))
        transform[:3, :3] = rot
        
        mesh.apply_transform(transform)
        vox = mesh.voxelized(pitch=2.0/128, method='binvox', bounds=np.array([[-1, -1, -1], [1, 1, 1]]), exact=True)
        vox.fill()
        vox = vox.matrix
        np.save(os.path.join(save_dir, mesh_name[:-4] + ('_%03d_' % (j*12)) + '.npy'), vox)
    # a = vox
    # s0 = np.sum(a, axis=0)
    # s1 = np.sum(a, axis=1)
    # s2 = np.sum(a, axis=2)
    # plt.subplot(131)
    # plt.imshow(s0)
    # plt.subplot(132)
    # plt.imshow(s1)
    # plt.subplot(133)
    # plt.imshow(s2)
    # plt.savefig('test.jpg')
    # exit(0)
        
    # verts, faces, normals, values = measure.marching_cubes_lewiner(vox, 0.5)
    # verts /= 128
    # verts[:, 0] -= 0.5
    # verts[:, 2] -= 0.5
    # save_obj_mesh('show/conv3d.obj', verts, faces)
    # exit(0)

