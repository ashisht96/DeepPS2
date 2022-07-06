import numpy as np 
import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.init import kaiming_normal_
import math
import pdb
import util
import warnings
warnings.filterwarnings("ignore")


class net_arch(nn.Module):
	def __init__(self, c_in, n_bins):

		super(net_arch, self).__init__()

		self.c_in = c_in
		self.n_bins = n_bins
		self.m = 3 #positional encoding

		self.encoder = init_encoder(self.c_in)
		self.normal_dec = decoder(task=1)
		self.diff_albedo_dec = decoder(task=2)
		self.light_net = lightNet(self.n_bins)
		self.refine_albedo = refAlbedo(self.m)
		self.recon_net = reconNet()
		self.relight_net = relightNet()

		for m in self.modules():
			if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
				kaiming_normal_(m.weight.data)
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	def forward(self, data):


		x = self.prepare_input(data)# x : batch of 8 imgs with mask (8,4,H,W)

		x1,x2,x3,x4,x5 = self.encoder(x)
		normal, norm_feat = self.normal_dec(x1,x2,x3,x4,x5)
		albedo = self.diff_albedo_dec(x1,x2,x3,x4,x5)
		l_theta, l_phi = self.light_net(torch.cat((normal, albedo),dim=1))
		dirs = self.get_actual_directions(l_theta, l_phi)
		ref_albedo, ntl = self.refine_albedo(dirs, x, normal, albedo)
		R, recon_imgs = self.recon_net(normal, ref_albedo, dirs, ntl)
		rel_img = self.relight_net(x, normal, dirs)

		# res = {'normal':normal, 'diff_albedo': albedo, 'spec_albedo': spec_albedo, 'ref_albedo': ref_albedo, 'l_theta': l_theta, 'l_phi':l_phi, 'R': R, 'recon_imgs': recon_imgs, 'rel_img': rel_img}
		res = {'normal':normal, 'diff_albedo': albedo, 'ref_albedo': ref_albedo, 'l_theta': l_theta, 'l_phi':l_phi, 'R': R, 'recon_imgs': recon_imgs, 'rel_img': rel_img}
		return res


	def prepare_input(self, data):
		img = data['img']
		mask = data['mask']
		final_input = torch.cat((img, mask),dim=1)
		return final_input

	def get_actual_directions(self, l_theta, l_phi):

		theta_cls = l_theta.argmax(dim=2).view(-1)
		phi_cls = l_phi.argmax(dim=2).view(-1)
		_, _, dirs = util.ClasstoDirs(theta_cls, phi_cls, self.n_bins)
		dirs = dirs.view(l_theta.size(0), l_theta.size(1), -1)

		return dirs



class init_encoder(nn.Module):
	def __init__(self, c_in):
		super(init_encoder, self).__init__()

		self.conv1 = nn.Conv2d(in_channels=c_in, out_channels=32, kernel_size=6,stride=2,padding=2, bias=False)
		self.bn1 = nn.BatchNorm2d(32)
		self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4,stride=2,padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(64)
		self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4,stride=2,padding=1, bias=False)
		self.bn3 = nn.BatchNorm2d(128)
		self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4,stride=2,padding=1, bias=False)
		self.bn4 = nn.BatchNorm2d(256)
		self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4,stride=2,padding=1, bias=False)
		self.bn5 = nn.BatchNorm2d(512)

	def forward(self, x):
		
		x1 = F.relu(self.bn1(self.conv1(x)), True)
		x2 = F.relu(self.bn2(self.conv2(x1)), True)
		x3 = F.relu(self.bn3(self.conv3(x2)), True)
		x4 = F.relu(self.bn4(self.conv4(x3)), True)
		x5 = F.relu(self.bn5(self.conv5(x4)), True)

		
		return x1,x2,x3,x4,x5

class interm_encoder(nn.Module):
	def __init__(self):
		super(interm_encoder, self).__init__()

		self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=6,stride=2,padding=2, bias=False)
		self.bn1 = nn.BatchNorm2d(32)
		self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4,stride=2,padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(64)
		self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4,stride=2,padding=1, bias=False)
		self.bn3 = nn.BatchNorm2d(128)
		self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4,stride=2,padding=1, bias=False)
		self.bn4 = nn.BatchNorm2d(256)
		self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4,stride=2,padding=1, bias=False)
		self.bn5 = nn.BatchNorm2d(512)

	def forward(self, x):

		x1 = F.relu(self.bn1(self.conv1(x)), True)
		x2 = F.relu(self.bn2(self.conv2(x1)), True)
		x3 = F.relu(self.bn3(self.conv3(x2)), True)
		x4 = F.relu(self.bn4(self.conv4(x3)), True)
		x5 = F.relu(self.bn5(self.conv5(x4)), True)

		return x1,x2,x3,x4,x5


class decoder(nn.Module):
	def __init__(self, task=1):
		super(decoder,self).__init__()

		############################ Tasks #####################################
		
		#Task 1: Normal
		#Task 2: Diffuse Albedo
		

		########################################################################

		self.dconv5 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False)
		self.bn5 = nn.BatchNorm2d(256)
		self.dconv4 = nn.ConvTranspose2d(in_channels=256*2, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False)
		self.bn4 = nn.BatchNorm2d(128)
		self.dconv3 = nn.ConvTranspose2d(in_channels=128*2, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False)
		self.bn3 = nn.BatchNorm2d(64)
		self.dconv2 = nn.ConvTranspose2d(in_channels=64*2, out_channels=32, kernel_size=4, stride=2, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(32)
		self.dconv1 = nn.ConvTranspose2d(in_channels=32*2, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(64)

		self.final_conv_norm = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=5, stride=1, padding=2, bias=True)
		self.final_conv_albedo = nn.Conv2d(in_channels=64, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True)
		self.task = task
		assert(task >=1 and task <=2)


	def forward(self, x1, x2, x3, x4, x5):
		
		xd1 = F.relu(self.bn5(self.dconv5(x5)), True)
		xd1 = torch.cat((xd1, x4), dim=1)
		xd2 = F.relu(self.bn4(self.dconv4(xd1)), True)
		xd2 = torch.cat((xd2, x3), dim=1)
		xd3 = F.relu(self.bn3(self.dconv3(xd2)), True)
		norm_feat = torch.cat((xd3, x2), dim=1)
		xd4 = F.relu(self.bn2(self.dconv2(norm_feat)), True)
		xd4 = torch.cat((xd4, x1), dim=1)
		xd5 = F.relu(self.bn1(self.dconv1(xd4)), True)
		
		

		if self.task == 1:
			x_final = torch.tanh(self.final_conv_norm(xd5))
			
			norm = torch.sqrt(torch.sum(x_final*x_final, dim=1).unsqueeze(1)).expand_as(x_final);
			x = x_final / norm
			return x, norm_feat

		elif self.task == 2:
			x_final = torch.tanh(self.final_conv_albedo(xd5))
			
			x = x_final
			return x
		


class lightNet(nn.Module):
	def __init__(self, n_bins):
		super(lightNet, self).__init__()

		self.conv1 = nn.Conv2d(in_channels=9, out_channels=64, kernel_size=3, stride=1, padding=0, bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0, bias=False)
		self.bn2 = nn.BatchNorm2d(128)
		self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0, bias=False)
		self.bn3 = nn.BatchNorm2d(256)
		self.n_bins = n_bins
		
		
		self.reg_theta = nn.Sequential(

			nn.Linear(256,256), 
			nn.ReLU(True), 
			nn.Dropout(0.25), 
			nn.Linear(256,64), 
			nn.ReLU(True), 
			nn.Dropout(0.25),
			nn.Linear(64, self.n_bins)

								)

		self.reg_phi = nn.Sequential(

			nn.Linear(256,256), 
			nn.ReLU(True), 
			nn.Dropout(0.25), 
			nn.Linear(256,64), 
			nn.ReLU(True), 
			nn.Dropout(0.25),
			nn.Linear(64, self.n_bins)

								)

	def forward(self, x):
		
	

		ldir = F.relu(self.bn1(self.conv1(x)))
		ldir = F.relu(self.bn2(self.conv2(ldir)))
		ldir = F.relu(self.bn3(self.conv3(ldir)))
		ldir = ldir.view(ldir.size(0), 2, ldir.size(1), -1)
		ldir = torch.mean(ldir, dim=3)		
		ldir_theta = self.reg_theta(ldir)
		ldir_phi = self.reg_phi(ldir)
		
		

		return ldir_theta, ldir_phi



class refAlbedo(nn.Module):
	def __init__(self, m):
		super(refAlbedo, self).__init__()
		self.m = m
		
		
		self.conv1 = nn.Conv2d(in_channels=44, out_channels=128, kernel_size=6,stride=2,padding=2, bias=False)
		self.bn1 = nn.BatchNorm2d(128)
		self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4,stride=2,padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(128)
		self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4,stride=2,padding=1, bias=False)
		self.bn3 = nn.BatchNorm2d(256)

		self.dconv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False)
		self.dbn3 = nn.BatchNorm2d(128)
		self.dconv2 = nn.ConvTranspose2d(in_channels=128*2, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False)
		self.dbn2 = nn.BatchNorm2d(128)
		self.dconv1 = nn.ConvTranspose2d(in_channels=128*2, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False)
		self.dbn1 = nn.BatchNorm2d(64)
		self.final_conv = nn.Conv2d(in_channels=64, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True)
		


	def positional_encoding(self,p,m):

		pi = torch.pi
		
		p_final= p
		for i in range(0,m):
			p_mod= 2**i*p*pi
			p_final = torch.cat((p_final, torch.sin(p_mod), torch.cos(p_mod)), dim=1)

		return p_final


	

	def forward(self, dirs, imgs, normal, albedo):

		

		N,C,H,W = normal.shape
		norm = normal.reshape(N,3,-1)

		v = torch.Tensor([[0,0,1]]).tile(2,1).unsqueeze(0).tile(N,1,1).cuda()	
		l = dirs
			
		denom = torch.sqrt(torch.sum((l+v)*(l+v), dim=2)).unsqueeze(2).tile(1,1,3)
		h = (l+v)/denom
		
		nth = torch.bmm(norm.transpose(1,2),h.transpose(1,2)).reshape(N,-1,H,W)
		vth = torch.bmm(v, h.transpose(1,2))[:,0,:].unsqueeze(2).unsqueeze(2).tile(1,1,H,W)

		ntl = torch.bmm(norm.transpose(1,2),l.transpose(1,2)).reshape(N,-1,H,W)
		ntl = torch.clamp(ntl, min=0.0)
		
		p = torch.cat((nth,vth),dim=1)
		p_pos = self.positional_encoding(p,self.m)	

		inp = torch.cat((p_pos, imgs, normal, albedo), dim=1)
	
		x1 = F.relu(self.bn1(self.conv1(inp)))
		x2 = F.relu(self.bn2(self.conv2(x1)))
		x3 = F.relu(self.bn3(self.conv3(x2)))

		x4 = F.relu(self.dbn3(self.dconv3(x3)))
		x4 = torch.cat((x4,x2), dim=1)
		x5 = F.relu(self.dbn2(self.dconv2(x4)))
		x5 = torch.cat((x5,x1), dim=1)
		x6 = F.relu(self.dbn1(self.dconv1(x5)))
		
		ref_albedo = torch.tanh(self.final_conv(x6))

		# ref_albedo = albedo + spec_albedo

		
		return ref_albedo, ntl



class reconNet(nn.Module):
	def __init__(self):
		super(reconNet, self).__init__()

		

		self.conv1 = nn.Conv2d(in_channels=15, out_channels=64, kernel_size=6,stride=2,padding=2, bias=False)
		self.bn1 = nn.BatchNorm2d(64)

		self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4,stride=2,padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(128)
		self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4,stride=2,padding=1, bias=False)
		self.bn3 = nn.BatchNorm2d(128)
		self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4,stride=2,padding=1, bias=False)
		self.bn4 = nn.BatchNorm2d(256)

		self.dconv4 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False)
		self.dbn4 = nn.BatchNorm2d(128)
		self.dconv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False)
		self.dbn3 = nn.BatchNorm2d(128)
		self.dconv2 = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False)
		self.dbn2 = nn.BatchNorm2d(64)
		self.dconv1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False)
		self.dbn1 = nn.BatchNorm2d(64)

		self.final_conv = nn.Conv2d(in_channels=64, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True)



            
		
	def forward(self, normal, albedo, dirs, ntl):

		N,C,H,W = normal.shape

		
		ldirs = dirs.view(N,-1).unsqueeze(2).unsqueeze(2).tile(1,1,H,W)
		x = torch.cat((normal, albedo, ldirs), dim=1)
		x1 = F.relu(self.bn1(self.conv1(x)))
		x2 = F.relu(self.bn2(self.conv2(x1)))
		x3 = F.relu(self.bn3(self.conv3(x2)))
		x4 = F.relu(self.bn4(self.conv4(x3)))

		x5 = F.relu(self.dbn4(self.dconv4(x4)))
		x5 = torch.cat((x5,x3), dim=1)
	
		
		x6 = F.relu(self.dbn3(self.dconv3(x5)))
		x6 = torch.cat((x6,x2), dim=1)
		x7 = F.relu(self.dbn2(self.dconv2(x6)))
		x7 = torch.cat((x7,x1), dim=1)
		x8 = F.relu(self.dbn1(self.dconv1(x7)))
		
		R = torch.tanh(self.final_conv(x8))

	
		ntl = torch.cat((ntl[:,:1,:,:].tile(1,3,1,1), ntl[:,1:,:,:].tile(1,3,1,1)), dim=1)

		recon_imgs = R*ntl

		return R, recon_imgs





class relightNet(nn.Module):
	def __init__(self):
		super(relightNet, self).__init__()



		self.conv1 = nn.Conv2d(in_channels=7, out_channels=64, kernel_size=6,stride=2,padding=2, bias=False)
		self.bn1 = nn.BatchNorm2d(64)

		self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4,stride=2,padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(128)
		self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4,stride=2,padding=1, bias=False)
		self.bn3 = nn.BatchNorm2d(128)
		self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4,stride=2,padding=1, bias=False)
		self.bn4 = nn.BatchNorm2d(256)

		self.dconv4 = nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False)
		self.dbn4 = nn.BatchNorm2d(128)
		self.dconv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False)
		self.dbn3 = nn.BatchNorm2d(128)
		self.dconv2 = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False)
		self.dbn2 = nn.BatchNorm2d(64)
		self.dconv1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False)
		self.dbn1 = nn.BatchNorm2d(64)

		self.final_conv = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=5, stride=1, padding=2, bias=True)


		self.light_feature = nn.Sequential(
											nn.Conv2d(in_channels = 3, out_channels=64 , kernel_size=1, stride=1, padding=0, bias=False),
											nn.Conv2d(in_channels = 64, out_channels=128 , kernel_size=1, stride=1, padding=0, bias=False),
											nn.BatchNorm2d(128),
											nn.Upsample(scale_factor=2, mode='bilinear'),
            								nn.Conv2d(in_channels = 128, out_channels=128 , kernel_size=3, stride=1, padding=1, bias=False),
            								nn.BatchNorm2d(128),
            								nn.Upsample(scale_factor=2, mode='bilinear'),
            								nn.Conv2d(in_channels = 128, out_channels=256 , kernel_size=3, stride=1, padding=1, bias=False),
            								nn.BatchNorm2d(256),
            								nn.Upsample(scale_factor=2, mode='bilinear'),
            								nn.Conv2d(in_channels = 256, out_channels=256 , kernel_size=3, stride=1, padding=1, bias=False),
            								nn.BatchNorm2d(256)
            								)

            
		
	def forward(self, img, normal, dirs):

		N,C,H,W = normal.shape

		inp_img = img[:,3:6,:,:]
		mask = img[:,-1:,:,:]

		ldir = dirs[:,0,:].view(N,-1,1,1)
		l_feat = self.light_feature(ldir)

		# import pdb
		# pdb.set_trace()
		
		x = torch.cat((inp_img, mask, normal), dim=1)

		x1 = F.relu(self.bn1(self.conv1(x)))
		x2 = F.relu(self.bn2(self.conv2(x1)))
		x3 = F.relu(self.bn3(self.conv3(x2)))
		x4 = F.relu(self.bn4(self.conv4(x3)))

		x4 = torch.cat((x4,l_feat), dim=1)
		x5 = F.relu(self.dbn4(self.dconv4(x4)))
		x5 = torch.cat((x5,x3), dim=1)
		x6 = F.relu(self.dbn3(self.dconv3(x5)))
		x6 = torch.cat((x6,x2), dim=1)
		x7 = F.relu(self.dbn2(self.dconv2(x6)))
		x7 = torch.cat((x7,x1), dim=1)
		x8 = F.relu(self.dbn1(self.dconv1(x7)))

		
		rel_img = torch.tanh(self.final_conv(x8))

		

		return rel_img

























