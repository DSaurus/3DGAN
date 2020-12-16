import torch
from torch.utils.data import Dataset
import os
import numpy as np
import matplotlib.pyplot as plt

class TrainDataset(Dataset):
    def __init__(self, dataroot):
        super(TrainDataset, self).__init__()
        self.dataroot = dataroot
        self.obj_list = os.listdir(self.dataroot)
    
    def __getitem__(self, index):
        obj_name = self.obj_list[index]
        res = {}
        res['name'] = obj_name
        res['vox'] = torch.FloatTensor(np.load(os.path.join(self.dataroot, obj_name)))
        
        # a = np.load(os.path.join(self.dataroot, obj_name))
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
        return res
    
    def __len__(self):
        return len(self.obj_list)
        
