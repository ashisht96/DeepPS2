import argparse
import os
import torch

class BaseOpts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    def initialize(self):
        #### Trainining Dataset ####
        self.parser.add_argument('--dataset',     default='DiLiGenT')
        self.parser.add_argument('--data_dir',    default='data/datasets/PS_Blobby_Dataset')

        
        #### Training Data and Preprocessing Arguments ####
        self.parser.add_argument('--rescale',     default=True,  action='store_false')
        self.parser.add_argument('--rand_sc',     default=True,  action='store_false')
        self.parser.add_argument('--scale_h',     default=128,   type=int)
        self.parser.add_argument('--scale_w',     default=128,   type=int)
        self.parser.add_argument('--crop',        default=True,  action='store_false')
        self.parser.add_argument('--crop_h',      default=128,   type=int)
        self.parser.add_argument('--crop_w',      default=128,   type=int)
        self.parser.add_argument('--test_h',      default=128,   type=int)
        self.parser.add_argument('--test_w',      default=128,   type=int)
        self.parser.add_argument('--retrain',     default=None)

        self.parser.add_argument('--cuda',        default=True,  action='store_false')
        self.parser.add_argument('--workers',     default=8,     type=int)

    
          
            
    def collectInfo(self):
        self.args.val_keys  = [
                'batch', 'scale_h', 'crop_h', 'init_lr', 'normal_w', 
                'dir_w', 'ints_w', 'in_img_num', 'dirs_cls', 'ints_cls'
                ]
        self.args.bool_keys = ['retrain'] 

    def parse(self):
        self.args = self.parser.parse_args()

        str_ids = self.args.gpu_ids.split(',')
        self.args.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.args.gpu_ids.append(id)

        if len(self.args.gpu_ids) > 0:
            torch.cuda.set_device(self.args.gpu_ids[0])
           
        return self.args
