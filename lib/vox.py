import trimesh
import os
import numpy as np

mesh_dir = '../dataset/dh/mesh/'
save_dir = '../dataset/dh/vox/'

for mesh_name in os.listdir(mesh_dir):
    mesh = trimesh.load(os.path.join(mesh_dir, mesh_name))

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
    mesh.apply_transfrom(transform)

    vox = mesh.voxelized(pitch=1.0/128, method='binvox', bounds=np.array([[-1, -1, -1], [1, 1, 1]]), exact=True)
    np.save(os.path.join(save_dir, mesh_name + '.npy'), vox)

