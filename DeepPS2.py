import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
from skimage import color
from L1_plus_perceptualLoss import L1_plus_perceptualLoss
import ms_loss
import cv2
import random
import models
import losses
import util
import torchvision.utils as vutils
from collections import OrderedDict
import numpy as np
import util


class DeepPS2(nn.Module):

	def __init__(self, args):
		super(DeepPS2,self).__init__()

		self.args = args	

		self.net = models.net_arch(self.args.c_in, self.args.n_bins).cuda(self.args.gpu_ids[0])		

		self.lighting_crit_theta = torch.nn.CrossEntropyLoss().cuda(self.args.gpu_ids)
		self.lighting_crit_phi = torch.nn.CrossEntropyLoss().cuda(self.args.gpu_ids)
		self.L1_loss_crit= torch.nn.L1Loss().cuda(self.args.gpu_ids[0])
		self.normal_crit = torch.nn.CosineEmbeddingLoss().cuda(self.args.gpu_ids[0])		
		self.color_crit = torch.nn.CosineSimilarity().cuda(self.args.gpu_ids[0])

		self.mse = torch.nn.MSELoss().cuda()
		self.ssim = ms_loss.MS_SSIM(max_val=1).cuda()

		self.optimizer = torch.optim.Adam(self.net.parameters(), self.args.init_lr, betas=(self.args.beta_1, self.args.beta_2))
		self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

		self.L1_perp_crit = L1_plus_perceptualLoss(self.args.lambda_A, self.args.lambda_B, self.args.perceptual_layers, self.args.gpu_ids, self.args.percep_is_l1, self.args.select_loss)

		if self.args.retrain:
			self.load_network(self.net, 'net', self.args,which_epoch) 

	def forward(self, data, epoch):
		self.data = data
		self.epoch = epoch
		self.res = self.net(self.data)

		self.tgt_phi, self.tgt_theta , self.pred_phi, self.pred_theta = self.get_lights(self.data, self.res)

		self.optimizer.zero_grad()
		self.backward()
		self.optimizer.step()



	def backward(self):

		self.total_loss = 0

		mask = self.data['mask']
		inp_img1 = self.data['img'][:,:3,:,:]*mask
		inp_img2 = self.data['img'][:,3:6,:,:]*mask
		recon_img1 = self.res['recon_imgs'][:,:3,:,:]*mask
		recon_img2 = self.res['recon_imgs'][:,3:6,:,:]*mask
		rel_img = self.res['rel_img']*mask



		self.img_recon_loss = self.L1_perp_crit(inp_img1, recon_img1) + self.L1_perp_crit(inp_img2, recon_img2)
		self.img_rel_loss = self.L1_perp_crit(inp_img1, rel_img)
		self.img_grad_loss = self.L1_loss_crit(self.gradient_image(inp_img1), self.gradient_image(recon_img1))+ self.L1_loss_crit(self.gradient_image(inp_img2), self.gradient_image(recon_img2)) + self.L1_loss_crit(self.gradient_image(inp_img1), self.gradient_image(rel_img))


		self.color_loss = (1.0 - self.color_crit(inp_img1, recon_img1).mean()) + (1.0 - self.color_crit(inp_img2, recon_img2).mean()) + (1.0 - self.color_crit(inp_img1, rel_img).mean())
		
		self.light_loss = self.lighting_crit_theta(self.pred_theta, self.tgt_theta) + self.lighting_crit_phi(self.pred_phi,self.tgt_phi)

		self.mse_img1 = self.mse(inp_img1, recon_img1)
		self.mse_img2 = self.mse(inp_img2, recon_img2)
		self.mse_rel = self.mse(inp_img1, recon_img1)

		self.ssim_img1 = self.ssim(inp_img1, recon_img1)
		self.ssim_img2 = self.ssim(inp_img2, recon_img2)
		self.ssim_rel = self.ssim(inp_img1, recon_img1)
		
		# self.smooth_const = self.args.alpha * losses.total_variation_loss(self.res['normal'], 'square') + self.args.beta * losses.total_variation_loss(self.res['ref_albedo'], 'absolute') + self.args.gamma * losses.total_variation_loss(self.res['diff_albedo'], 'absolute')

		
		self.total_loss += (self.args.lambda_img * (self.img_recon_loss + self.img_rel_loss + self.img_grad_loss + self.color_loss) + self.args.lambda_light*self.light_loss)

		if self.epoch<25:
			n_est, n_tar = self.res['normal'], self.data['normal']
			n_num = n_tar.nelement() // n_tar.shape[1]
			if not hasattr(self, 'n_flag') or n_num != self.n_flag.nelement():
			    self.n_flag = n_tar.data.new().resize_(n_num).fill_(1)
			self.out_reshape = n_est.permute(0, 2, 3, 1).contiguous().view(-1, 3)
			self.gt_reshape  = n_tar.permute(0, 2, 3, 1).contiguous().view(-1, 3)
			self.normal_loss = self.normal_crit(self.out_reshape, self.gt_reshape, self.n_flag)

			self.albedo_loss = self.L1_perp_crit(self.data['diff_albedo']*self.data['mask'], self.res['diff_albedo']*self.data['mask']) + self.L1_perp_crit(self.data['albedo']*self.data['mask'], self.res['ref_albedo']*self.data['mask'])


			
			self.total_loss += (self.normal_loss + self.albedo_loss)

		self.total_loss.backward()		


	def get_lights(self, tgt, pred):

		l_tgt = tgt['l_dirs'][:,1,:]
		
		l_phi, l_theta = util.DirsToClass(l_tgt, self.args.n_bins)
		l_phi_tgt = torch.zeros(l_phi.shape[0],self.args.n_bins).cuda()
		l_theta_tgt = torch.zeros(l_theta.shape[0],self.args.n_bins).cuda()
		l_phi_tgt[torch.arange(l_phi.shape[0]), l_phi.long()] = 1
		l_theta_tgt[torch.arange(l_theta.shape[0]), l_theta.long()] = 1

		
		l_phi_pred = pred['l_phi'][:,1,:]
		l_theta_pred = pred['l_phi'][:,1,:]

		

		return l_phi_tgt, l_theta_tgt, l_phi_pred, l_theta_pred

	def gradient_image(self, im):
		sobel_x = torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).repeat(1, 3, 1, 1).cuda()
		sobel_y = torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).repeat(1, 3, 1, 1).cuda()
		gx = F.conv2d(im, sobel_x, padding=1)
		gy = F.conv2d(im, sobel_y, padding=1)
		grad = torch.abs(gx) + torch.abs(gy)
		d, _ = grad.view(im.shape[0], -1).max(1)
		d = d.resize(im.shape[0], 1, 1, 1)
		grad = grad/d
		return grad



	def load_network(self, network, network_label, epoch_label):
		save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
		save_path = os.path.join(self.args.checkpoints_dir, save_filename)
		network.load_state_dict(torch.load(save_path))

	def save(self, label):
		self.save_network(self.net, 'net', label, self.args.gpu_ids)     


	def update_learning_rate(self):
		self.scheduler.step()
		lr = self.optimizer.param_groups[0]['lr']
		print('learning rate = %.7f' % lr)


	def calSiMse(self, gt_img, pred_img):
	    scale = torch.gels(gt_img.view(-1, 1), pred_img.view(-1, 1))
	    error = ((gt_img - scale[0][0]*pred_img)**2).mean()
	    return error




	def get_results(self, epoch, iters):

		mask = self.data['mask'].data
		img1 = self.data['img'][:,:3,:,:]*mask.data		
		img2 = self.data['img'][:,3:6,:,:]*mask.data 

		recon_img1 = self.res['recon_imgs'][:,:3,:,:]*mask.data + (1-mask)
		recon_img2 = self.res['recon_imgs'][:,3:6,:,:]*mask.data + (1-mask)

		
		

		gt_normal = self.data['normal'].data
		pred_normal = self.res['normal'].data

		pred_diff_albedo1 = self.res['diff_albedo'][:,:3,:,:]*mask.data 
		pred_refine_albedo1 = self.res['ref_albedo'][:,:3,:,:]*mask.data 
		gt_diff_albedo1 = self.data['diff_albedo'][:,:3,:,:]*mask.data 
		gt_albedo1 = self.data['albedo'][:,:3,:,:]*mask.data 


		pred_diff_albedo2 = self.res['diff_albedo'][:,3:,:,:]*mask.data 
		pred_refine_albedo2 = self.res['ref_albedo'][:,3:,:,:]*mask.data 
		gt_diff_albedo2 = self.data['diff_albedo'][:,3:,:,:]*mask.data 
		gt_albedo2 = self.data['albedo'][:,3:,:,:]*mask.data 



		rel_img = self.res['rel_img']*mask.data


		
		
		if self.args.have_gt_n:
		    acc, error_map = util.calNormalAcc(gt_normal, pred_normal, mask)
		    pred_n = (pred_normal + 1) / 2
		    masked_pred = pred_n * mask.expand_as(pred_normal)
		    final_pred_normal = [masked_pred.cpu() + (1-mask).cpu(), error_map['angular_map'].cpu() + (1-mask).cpu()]
		    
		results = [img1 , ((gt_normal + 1) / 2)*mask + (1-mask), error_map['angular_map'].cpu() + (1-mask).cpu(), gt_diff_albedo1, pred_diff_albedo1, gt_albedo1, pred_refine_albedo1, recon_img1, masked_pred,
					img2,  masked_pred.cpu() + (1-mask).cpu(), rel_img, gt_diff_albedo2, pred_diff_albedo2, gt_albedo2, pred_refine_albedo2, recon_img2, error_map['angular_map']] 

		res = [torch.unsqueeze(item[0], 0).cpu() for item in results]
		res = torch.cat(self.convertToSameSize(res))
		save_prefix = os.path.join(self.args.save_dir, self.args.obj, 'epoch_%d.png' % epoch)
		vutils.save_image(res, save_prefix , nrow=9, padding=0)


		loss_box = OrderedDict([('total_loss', self.total_loss)])

		if np.isnan(loss_box['total_loss'].detach().cpu()):
		    print('---------Loss is NaN-----------')
		    sys.exit(1)

		# loss_box['geometric_constraint'] = self.geo_loss
		# loss_box['smoothness_constraint'] = self.smooth_const
		loss_box['img_recon_loss'] = self.img_recon_loss
		loss_box['img_rel_loss'] = self.img_rel_loss
		loss_box['img_grad_loss'] = self.img_grad_loss
		loss_box['color_loss'] = self.color_loss
		loss_box['light_loss'] = self.light_loss
		loss_box['mae_normal'] = acc['n_err_mean']
		loss_box['mse_img1'] = self.mse_img1
		loss_box['mse_img2'] = self.mse_img2
		loss_box['mse_rel_img'] = self.mse_rel
		loss_box['ssim_img1'] = self.ssim_img1
		loss_box['ssim_img2'] = self.ssim_img2
		loss_box['ssim_rel_img'] =self.ssim_rel

		

		if self.epoch<15:
			loss_box['normal_loss'] = self.normal_loss
			loss_box['albedo_loss'] = self.albedo_loss

		message = '(epoch: %d, iters: %d) ' % (epoch, iters)
		for k, v in loss_box.items():
		    message += '%s: %.3f ' % (k, v)
		print(message)
	        


	def convertToSameSize(self, t_list):
		shape = (t_list[0].shape[0], 3, t_list[0].shape[2], t_list[0].shape[3])
		for i, tensor in enumerate(t_list):
		    n, c, h, w = tensor.shape
		    if tensor.shape[1] != shape[1]: # check channel
		        t_list[i] = tensor.expand((n, shape[1], h, w))
		    if h == shape[2] and w == shape[3]:
		        continue
		    t_list[i] = F.upsample(tensor, [shape[2], shape[3]], mode='bilinear')
		return t_list



	def save_network(self, network, network_label, epoch_label, gpu_ids):
		save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
		save_path = os.path.join(self.args.checkpoints_dir, self.args.obj, save_filename)
		torch.save(network.cpu().state_dict(), save_path)
		if len(gpu_ids) and torch.cuda.is_available():
			network.cuda()