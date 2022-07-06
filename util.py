
import os
import numpy as np
import torch
from skimage.transform import resize
import math
from matplotlib import cm


def readList(list_path):
    lists = []
    with open(list_path) as f:
        lists = f.read().splitlines()
    return lists


def rescale(inputs, size):
    in_h, in_w, _ = inputs.shape
    h, w = size
    if h != in_h or w != in_w:
        inputs = resize(inputs, size, order=1, mode='reflect')   
    return inputs


def arrayToTensor(array):
    if array is None:
        return array

    array = np.transpose(array, (2, 0, 1))
    tensor = torch.from_numpy(array)
    return tensor.float()

def convert2Tensor(item):
	if item is None:
		return item

	item = [arrayToTensor(k) for k in item]
	item = torch.cat(item,0)
	return item



def imgSizeToFactorOfK(img, k):
    if img.shape[0] % k == 0 and img.shape[1] % k == 0:
        return img
    pad_h, pad_w = k - img.shape[0] % k, k - img.shape[1] % k
    img = np.pad(img, ((0, pad_h), (0, pad_w), (0,0)), 
            'constant', constant_values=((0,0),(0,0),(0,0)))
    return img


def calNormalAcc(gt_n, pred_n, mask=None):
    """Tensor Dim: NxCxHxW"""
    # import pdb
    # pdb.set_trace()
    dot_product = (gt_n * pred_n).sum(1).clamp(-1,1)
    error_map   = torch.acos(dot_product) # [-pi, pi]
    angular_map = error_map * 180.0 / math.pi
    angular_map = angular_map * mask.narrow(1, 0, 1).squeeze(1)

    valid = mask.narrow(1, 0, 1).sum()
    ang_valid  = angular_map[mask.narrow(1, 0, 1).squeeze(1).byte()]
    n_err_mean = ang_valid.sum() / valid
    n_err_med  = ang_valid.median()
    n_acc_11   = (ang_valid < 11.25).sum().float() / valid
    n_acc_30   = (ang_valid < 30).sum().float() / valid
    n_acc_45   = (ang_valid < 45).sum().float() / valid

    angular_map = colorMap(angular_map.cpu().squeeze(1))
    value = {'n_err_mean': n_err_mean.item(), 
            'n_acc_11': n_acc_11.item(), 'n_acc_30': n_acc_30.item(), 'n_acc_45': n_acc_45.item()}
    angular_error_map = {'angular_map': angular_map}
    return value, angular_error_map

def colorMap(diff):
	thres = 90
	diff_norm = np.clip(diff, 0, thres) / thres
	diff_cm = torch.from_numpy(cm.jet(diff_norm.numpy()))[:,:,:, :3]
	return diff_cm.permute(0,3,1,2).clone().float()


def DirsToClass(dirs, cls_num):
    # dirs = dirs / dirs.norm(p=2, dim=1, keepdim=True)
    phi = torch.atan(dirs[:,2] / (dirs[:,0] + 1e-8)) 
    phi[torch.where(dirs[:,0]<0)] += torch.pi
    denom = torch.sqrt(dirs[:,0] * dirs[:,0] + dirs[:,1] * dirs[:,1] + dirs[:,2] * dirs[:,2])
    theta = torch.asin(dirs[:,1]/ (denom + 1e-8))
    theta = theta / np.pi * 180
    phi   = phi / np.pi * 180
    elevate = ((theta + 90.0) / 180 * cls_num).clamp(0, cls_num-1).long()
    azimuth = (phi / 180 * cls_num).clamp(0, cls_num-1).long()
    return azimuth, elevate


def CartesiantoSpherical(dirs):
    dirs = dirs / dirs.norm(p=2, dim=1, keepdim=True)
    phi = torch.atan(dirs[:,2] / (dirs[:,0] + 1e-8)) 
    phi[torch.where(dirs[:,0]<0)] += torch.pi
    
    denom = torch.sqrt(dirs[:,0] * dirs[:,0] + dirs[:,1] * dirs[:,1] + dirs[:,2] * dirs[:,2])
    theta = torch.asin(dirs[:,1]/ (denom + 1e-8))
    theta = theta / np.pi * 180
    phi   = phi / np.pi * 180

    return theta, phi

def SphericaltoCart(dirs):
    theta = dirs[:,0]* torch.pi / 180
    phi = dirs[:,1]* torch.pi / 180
    
    x = torch.cos(theta) * torch.cos(phi)
    y = torch.sin(theta)
    z = torch.cos(theta) * torch.sin(phi)

    res = torch.stack([x,y,z],1)

    return res

def ClasstoDirs(theta_cls, phi_cls, cls_num):
    div = 180 / cls_num
    theta = (( div / 2 ) - 90) +  div * theta_cls
    phi = (div / 2) + div * phi_cls
    
    dirs = SphericaltoCart(torch.stack([theta, phi], 1))

    return theta, phi, dirs