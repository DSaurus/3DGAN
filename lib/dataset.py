import torch
from torch.utils.data import Dataset
import os
import numpy as np

class TrainDatasetNormal(Dataset):
    def __init__(self, dataroot):
        super(TrainDatasetNormal, self).__init__()
        self.dataroot = dataroot
        self.obj_list = os.listdir(self.dataroot)
    
    def __getitem__(self, index):
        obj_name = self.obj_list[index]
        res = {}
        res['name'] = obj_name
        res['vox'] = torch.FloatTensor(np.load(os.path.join(self.dataroot, obj_name)))
        return res
        
