import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def total_variation_loss(img, loss_type):
	bs_img, c_img, h_img, w_img = img.size()

	if loss_type == 'absolute':
		tv_h = torch.abs(img[:,:,1:,:]-img[:,:,:-1,:]).sum()
		tv_w = torch.abs(img[:,:,:,1:]-img[:,:,:,:-1]).sum()

	elif loss_type == 'square':
		tv_h = torch.pow(img[:,:,1:,:]-img[:,:,:-1,:],2).sum()
		tv_w = torch.pow(img[:,:,:,1:]-img[:,:,:,:-1],2).sum()

	return (tv_h + tv_w) / (bs_img*c_img*h_img*w_img)

def geometric_loss(normal, depth):

	d_depth = depth_diff(depth)
	geo_loss = (normal*d_depth).sum(1).unsqueeze(1)
	geo_loss = torch.sum(1.0-geo_loss)/d_depth.numel()

	return geo_loss


def depth_diff(im):
	
	sobel_x = torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).unsqueeze(0).unsqueeze(0).cuda()
	sobel_y = torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).unsqueeze(0).unsqueeze(0).cuda()
	gx = F.conv2d(im, sobel_x, padding=1)
	gy = F.conv2d(im, sobel_y, padding=1)
	g = 1.0*torch.ones(gx.size()).cuda()
	depth_grad = torch.cat((-gx, -gy, g), dim=1)
	denom = torch.sqrt(torch.sum(depth_grad*depth_grad, dim=1)).unsqueeze(1)
	depth_grad = depth_grad/denom
	
	return depth_grad