import torch
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
import numpy as np
import os
from skimage import measure

from lib.dataset import TrainDataset
from lib.options import BaseOptions
from lib.compression import Compression2D
from utils.mesh_util import *


config = BaseOptions().parse()

os.makedirs(os.path.join(config.checkpoints_path, config.name), exist_ok=True)

if __name__ == "__main__":
    dataset = TrainDataset(config.dataroot)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_threads)
    log = SummaryWriter(config.log_path)

    gpu_ids = [int(i) for i in config.gpu_ids.split(',')]
    cuda = torch.device('cuda:%s' % config.gpu_ids[0])
    
    netG = Compression2D()
    netG.to(cuda)
    netG = DataParallel(netG, gpu_ids)

    if config.load_netG_checkpoint_path:
        print('loading from ' + config.load_netG_checkpoint_path)
        netG.load_state_dict(torch.load(config.load_netG_checkpoint_path))

    optimG = torch.optim.Adam(netG.parameters(), lr=1e-4)

    EPOCH = 10000
    accuracy = 0.0
    loss_f = torch.nn.MSELoss()
    for epoch in range(EPOCH):
        np.random.seed(int(time.time()))
        accuracy_sum = 0.0
        train_bar = tqdm(enumerate(dataloader))
        for train_idx, data in train_bar:
            vox = data["vox"].float().to(cuda)
            gen_vox = netG.forward(vox)
            sum1 = torch.sum(vox)
            sum0 = torch.sum(1-vox)
            loss = loss_f(gen_vox * sum0 / sum1, vox * sum0 / sum1)
            optimG.zero_grad()
            loss.backward()
            optimG.step()

            show_vox = gen_vox[0].detach().cpu().numpy()
            print(torch.max(gen_vox), torch.mean(gen_vox), torch.mean(vox))
            verts, faces, normals, values = measure.marching_cubes_lewiner(show_vox, 0.5)
            verts /= 128
            verts[:, 0] -= 0.5
            verts[:, 2] -= 0.5
            save_obj_mesh('show/conv3d.obj', verts, faces)
            exit(0)
            
            train_bar.set_description('loss : %f' % loss.item())
            log.add_scalar('loss', loss.item())

            if train_idx % config.freq_save == 0 and train_idx != 0:
                torch.save(netG.state_dict(), os.path.join(config.checkpoints_path, config.name, 'netG_latest'))
                torch.save(netG.state_dict(), os.path.join(config.checkpoints_path, config.name, 'netG_epoch_%d' % epoch))


