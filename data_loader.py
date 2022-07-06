from __future__ import division
import os
import numpy as np
import scipy.io as sio
from imageio import imread
from itertools import permutations
import random
import torch
import torch.utils.data as data
import util
import itertools
import pdb
import cv2

np.random.seed(0)

class load_dataset(data.Dataset):
    def __init__(self, args):
        self.root = './data/DiLiGenT'
        self.args = args
        self.obj_name = self.args.obj
        self.dir = os.path.join(self.root, self.obj_name)
        self.ldirs = np.genfromtxt(os.path.join(self.dir, 'light_directions.txt'))   
        
        
        self.normal = np.array(self._getNormal())
        self.mask = np.array(self._getMask())
        
        self.img_names = util.readList(os.path.join(self.dir, 'filenames.txt'))  

        self.img_list = []
        self.light_dirs = []

        self.meta_data = self.get_proxy_features(self.normal, self.ldirs)       

        self.spec_albedo = []
        self.diff_albedo = []
        self.albedo = []

        imgs = []
        spec_albs = []
        diff_albs = []
        albs = []
        idx = 0
        for name in self.img_names: 
            img = self._getImage(name, self.mask) 
            spec_alb =  img*np.expand_dims(self.meta_data['spec_sd'][:,:,idx],axis=2)
            alb =  img*np.expand_dims(self.meta_data['shading'][:,:,idx],axis=2)   
            diff_alb = alb-spec_alb
            imgs.append(img)
            spec_albs.append(spec_alb)
            diff_albs.append(diff_alb)
            albs.append(alb)
            idx+=1
    


        idx_list = list(itertools.permutations(range(self.ldirs.shape[0]),2))
        random.shuffle(idx_list)


        

        for inds in idx_list:
            self.img_list.append(np.concatenate((imgs[inds[0]], imgs[inds[1]]),axis=2))
            self.light_dirs.append(np.concatenate((self.ldirs[inds[0]].reshape(1,-1), self.ldirs[inds[1]].reshape(1,-1)),axis=0))
            self.spec_albedo.append(np.concatenate((spec_albs[inds[0]], spec_albs[inds[1]]), axis=2))
            self.diff_albedo.append(np.concatenate((diff_albs[inds[0]], diff_albs[inds[1]]), axis=2))
            self.albedo.append(np.concatenate((albs[inds[0]], albs[inds[1]]), axis=2))

         

        self.img_list = np.array(self.img_list)
        self.light_dirs = np.array(self.light_dirs)
        self.spec_albedo = np.array(self.spec_albedo)
        self.diff_albedo = np.array(self.diff_albedo)  

        

        
    def _getImage(self,img_name, mask):

        img_path = os.path.join(self.dir,img_name)
        img = cv2.imread(img_path).astype(np.float32) / 255.0
        img = util.rescale(img, [128,128])
        img = img * mask.repeat(3, 2)
      

        return img

    def _getNormal(self):
        normal_path = os.path.join(self.root, self.obj_name, 'Normal_gt.mat') 
        normal = sio.loadmat(normal_path)['Normal_gt']

        # Use 'Normal_init.mat' instead after creating the initial normal using norm_init.py

        normal = util.rescale(normal, [128,128])
        norm = np.sqrt((normal * normal).sum(2, keepdims=True))
        normal = normal / (norm + 1e-10)  
       
        return normal


    
    def _getMask(self):
        mask = cv2.imread(os.path.join(self.root, self.obj_name, 'mask.png'))
        if mask.ndim > 2: mask = mask[:,:,0]
        mask = mask.reshape(mask.shape[0], mask.shape[1], 1)/255.0
        mask = util.rescale(mask, [128,128])
        
        return mask


    def get_proxy_features(self,normal, dirs):
        item = {}
        n_new = normal
        # attached shadow
        # item['shadow'] = self._get_shadow(n_new, dirs).astype(np.float32)
        # specular shading
        item['spec_sd'] = self._get_spec_shading(n_new, dirs).astype(np.float32)
        # shading
        item['shading'] = self._get_shading(n_new, dirs).astype(np.float32)

        return item

    def normalize_to_unit_len(self,matrix, dim=2):
        denorm = np.sqrt((matrix * matrix).sum(dim, keepdims=True))
        matrix = matrix / (denorm + 1e-10)
        return matrix

    def _get_shadow(self,normal, dirs):
        h, w, c = normal.shape
        shadow = np.dot(normal.reshape(h * w, 3), dirs.transpose()) <= 0
        mask = normalToMask(normal)
        shadow = shadow.reshape(h, w, -1) * mask
        mx = shadow.max(0).max(0)
        mn = shadow.min(0).min(0)
        shadow = (255.0 / (mx - mn) * (shadow - mn))/255.0
        return shadow

    def _get_shading(self,normal, dirs):
        h, w, c = normal.shape
        shading = np.dot(normal.reshape(-1, 3), dirs.transpose()).clip(0)
        shading = shading.reshape(h, w, -1)
        mx = shading.max(0).max(0)
        mn = shading.min(0).min(0)
        shading = (255.0 / (mx - mn) * (shading - mn))/255.0
        return shading

    def _get_spec_shading(self,normal, dirs):
        # pdb.set_trace()
        h, w, c = normal.shape
        view = np.zeros(dirs.shape).astype(np.float32)
        view[:,2] = 1.0
        bisector = self.normalize_to_unit_len(view + dirs, dim=1)
        spec_shading = (np.dot(normal.reshape(-1, 3), bisector.transpose()).clip(0).reshape(h, w, -1))**2000
        mx = spec_shading.max(0).max(0)
        mn = spec_shading.min(0).min(0)
        spec_shading = (255.0 / (mx - mn) * (spec_shading - mn))/255.0
        return spec_shading

    
    def __getitem__(self, index):

        np.random.seed(index)
        
        item = {}

                
        imgs = util.arrayToTensor(self.img_list[index])         
        mask = util.arrayToTensor(self.mask) 
        normal = util.arrayToTensor(self.normal)
        spec_albedos = util.arrayToTensor(self.spec_albedo[index]) 
        diff_albedos = util.arrayToTensor(self.diff_albedo[index]) 
        albedos = util.arrayToTensor(self.albedo[index])
        light_direcs = torch.from_numpy(self.light_dirs[index])
        


        item = {'img': imgs, 'mask':mask, 'normal': normal, 'l_dirs':light_direcs, 'spec_albedo': spec_albedos, 'diff_albedo': diff_albedos, 'albedo':albedos}

        return item


    def __len__(self):
        return len(self.img_list)
