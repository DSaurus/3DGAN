import torch
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter

from lib.dataset import Dataset
from lib.options import BaseOptions
from lib.generator import Generator
from lib.discriminator import Discriminator


config = BaseOptions().parse()

if __name__ == "__main__":
    dataset = Dataset(config.dataroot)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    log = SummaryWriter(config.log_path)

    gpu_ids = config.gpu_ids
    netG = Generator()
    netD = Discriminator()
    netG = DataParallel(netG, gpu_ids)
    netD = DataParallel(netD, gpu_ids)
