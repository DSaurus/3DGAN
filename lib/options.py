import argparse
import os

class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # Datasets related
        g_data = parser.add_argument_group('Data')
        
        g_data.add_argument('--log_path', type=str, default='./train_log')
        g_data.add_argument('--dataroot', type=str, default='./data',
                            help='path to images (data folder)')
    
        # Experiment related
        g_exp = parser.add_argument_group('Experiment')
        g_exp.add_argument('--name', type=str, default='example',
                           help='name of the experiment. It decides where to store samples and models')
       
        # Training related
        g_train = parser.add_argument_group('Training')
        g_train.add_argument('--gpu_id', type=int, default=0, help='gpu id for cuda')
        g_train.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2, -1 for CPU mode')

        g_train.add_argument('--num_threads', default=1, type=int, help='# sthreads for loading data')
    
        g_train.add_argument('--batch_size', type=int, default=2, help='input batch size')
        g_train.add_argument('--learning_rate', type=float, default=1e-3, help='adam learning rate')
        g_train.add_argument('--num_epoch', type=int, default=1000, help='num epoch to train')

        g_train.add_argument('--freq_save', type=int, default=50, help='freqency of the save_checkpoints')
        g_train.add_argument('--train_2d', action='store_true')
        
        # path
        parser.add_argument('--checkpoints_path', type=str, default='./checkpoints', help='path to save checkpoints')
        parser.add_argument('--load_netG_checkpoint_path', type=str, default=None, help='path to load checkpoints')
        parser.add_argument('--load_netD_checkpoint_path', type=str, default=None, help='path to load checkpoints')
        parser.add_argument('--results_path', type=str, default='./results', help='path to save results ply')
        
        # special tasks
        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def parse(self):
        opt = self.gather_options()
        return opt
