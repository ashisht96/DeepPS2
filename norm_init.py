import numpy as np
from numpy import *
from scipy.io import loadmat, savemat
import cv2
import glob
import os
from scipy.sparse.linalg import spsolve

obj_name = 'ball'
im_path = os.path.join('data/DiLiGenT', obj_name)
loi = glob.glob(os.path.join(im_path,'*.png'))
loi.remove(os.path.join(im_path,'Normal_gt.png'))
loi.remove(os.path.join(im_path, 'mask.png'))

I = []
for idx in range(len(loi)):
	img = cv2.imread(loi[idx])
	img = np.expand_dims(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),2)
	I.append(img)
I = np.concatenate(I,2)

mask = cv2.imread(os.path.join(im_path,'mask.png'))
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)


L = np.array(np.genfromtxt(os.path.join(im_path, 'light_directions.txt')))

I_all = I.reshape(I.shape[0]*I.shape[1],I.shape[2]).T

M = linalg.pinv(L).dot(I_all)
M_1 = M[0]
M_2 = M[1]
M_3 = M[2]

rho = sqrt(M_1**2 + M_2**2 + M_3**2).reshape(I.shape[0],I.shape[1]) + 1e-6

n_1 = np.expand_dims((M_1 / rho.reshape(-1)).reshape(I.shape[0],I.shape[1]),2)
n_2 = np.expand_dims((M_2 / rho.reshape(-1)).reshape(I.shape[0],I.shape[1]),2)
n_3 = np.expand_dims((M_3 / rho.reshape(-1)).reshape(I.shape[0],I.shape[1]),2)

N = np.concatenate((n_1, n_2, n_3), axis=2)
Norm_init = {'Normal_init':N}
savemat(os.path.join(im_path,'Normal_init.mat'), Norm_init)



# N1 = loadmat(os.path.join(im_path, 'Normal_init.mat'))['Normal_init']
# print(N1.shape)