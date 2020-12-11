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
from lib.generator import Generator
from lib.discriminator import Discriminator
from utils.mesh_util import *


config = BaseOptions().parse()

os.makedirs(os.path.join(config.checkpoints_path, config.name), exist_ok=True)

if __name__ == "__main__":
    dataset = TrainDataset(config.dataroot)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_threads)
    log = SummaryWriter(config.log_path)

    gpu_ids = [int(i) for i in config.gpu_ids.split(',')]
    cuda = torch.device('cuda:%s' % config.gpu_ids[0])
    netG = Generator()
    netD = Discriminator()
    netG.to(cuda)
    netD.to(cuda)
    netG = DataParallel(netG, gpu_ids)
    netD = DataParallel(netD, gpu_ids)

    if config.load_netG_checkpoint_path:
        print('loading from ' + config.load_netG_checkpoint_path)
        netG.load_state_dict(torch.load(config.load_netG_checkpoint_path))
    if config.load_netD_checkpoint_path:
        print('loading from ' + config.load_netD_checkpoint_path)
        netD.load_state_dict(torch.load(config.load_netD_checkpoint_path))

    optimG = torch.optim.Adam(netG.parameters(), lr=1e-4)
    optimD = torch.optim.Adam(netD.parameters(), lr=1e-4)

    EPOCH = 100
    accuracy = 0.0
    for epoch in range(EPOCH):
        np.random.seed(int(time.time()))
        accuracy_sum = 0.0
        train_bar = tqdm(enumerate(dataloader))
        for train_idx, data in train_bar:
            vox = data["vox"].unsqueeze(1).to(cuda)
            random_vec = torch.FloatTensor(np.random.random((config.batch_size, 512, 1, 1, 1))).to(cuda)
            with torch.no_grad():
                gen_res = netG.forward(random_vec)
            verts, faces, normals, values = measure.marching_cubes_lewiner(gen_res[0].squeeze(0).detach().cpu().numpy(), 0.5)
            
            dis_gen = netD.forward(gen_res)
            dis_gt = netD.forward(vox)
            lossD = -torch.mean(torch.log(dis_gt)) - torch.mean(torch.log(1 - dis_gen))
            
            optimD.zero_grad()
            lossD.backward()
            if accuracy < 0.8:
                optimD.step()
            else:
                print('not train discriminator!')

                # verts, faces, normals, values = measure.marching_cubes_lewiner(gen_res[0].squeeze(0).detach().cpu().numpy(), 0.5)
                # verts, faces, normals, values = measure.marching_cubes_lewiner(vox[0].squeeze(0).detach().cpu().numpy(), 0.5)
                # print(gen_res[0].squeeze(0).detach().cpu().numpy())
                # print(vox[0])
                # verts /= 128
                # verts[:, 0] -= 0.5
                # verts[:, 2] -= 0.5
                # save_obj_mesh('show/conv3d.obj', verts, faces)
                # exit(0)

            gen_res = netG.forward(random_vec)
            dis_gen = netD.forward(gen_res)
            lossG = -torch.mean(torch.log(dis_gen))
            optimG.zero_grad()
            lossG.backward()
            optimG.step()
            
            # accuracy_sum += torch.sum(1-dis_gen) + torch.sum(dis_gt)
            # accuracy = accuracy_sum / (2*config.batch_size*(train_idx+1))
            accuracy = torch.mean(1-dis_gen) / 2 + torch.mean(dis_gt) / 2
            train_bar.set_description('discriminator accuracy : %f' % accuracy)
            log.add_scalar('gen_loss', lossG.item())
            log.add_scalar('dis_loss', lossD.item())

            if train_idx % config.freq_save == 0 and train_idx != 0:
                torch.save(netG.state_dict(), os.path.join(config.checkpoints_path, config.name, 'netG_latest'))
                torch.save(netG.state_dict(), os.path.join(config.checkpoints_path, config.name, 'netG_epoch_%d' % epoch))
                
                torch.save(netD.state_dict(), os.path.join(config.checkpoints_path, config.name, 'netD_latest'))
                torch.save(netD.state_dict(), os.path.join(config.checkpoints_path, config.name, 'netD_epoch_%d' % epoch))


