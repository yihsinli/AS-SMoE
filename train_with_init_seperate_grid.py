#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 08:54:51 2021

@author: yhl
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 13:34:18 2021

@author: yhl
"""
# python SMoE_overlap_batch_lssim_lg_test.py --u_bits 4 --a_bits 4 --B_bits 4 --block --steer --bz 64 --batch_size 4 --data_name lena_crop.jpg --image_path ../images --device_id -1 --lr 0.001 --n_epoch 10000 --K 25 --B_max 1000000 --border mirror --ol 3 --lssim 0.001 --lg 0 --load
# python SMoE_overlap_batch_lssim_lg.py --u_bits 4 --a_bits 4 --B_bits 4 --steer --bz 512 --batch_size 1 --data_name lena.tif --image_path ../images --device_id -1 --lr 0.001 --n_epoch 10000 --K 100 --B_max 1000000 --border mirror --ol 3 --lssim 0.001 --lg 0 --load --init_path ../SMoE_INIT/init_loc/lena_color_512_init.npy

import torch
import copy
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
import os
import argparse

from torchvision import transforms as T
from PIL import Image, ImageOps
from datetime import datetime
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from piqa import SSIM
import pytorch_ssim
#from AE import AE
from scipy.fftpack import dct, idct
from sklearn.cluster import KMeans
import glob
import scipy.io as sio
import pickle

parser = argparse.ArgumentParser()
#parser.add_argument('--name', type=str, default="img_lena_crop_2")
parser.add_argument('--data_path', type=str, default="./images/32x32/")
parser.add_argument('--map_path', type=str, default="./data/train/semantic/maps")
parser.add_argument('--result_path', type=str, default="model_weights")
parser.add_argument('--init_para_path', type=str, default="init_para")

parser.add_argument('--steer' , default=False,action='store_true')
parser.add_argument('--single', default=False,action='store_true')

parser.add_argument('--load'  , default=False,action='store_true')

parser.add_argument('--u_bits', type=int, default=0)
parser.add_argument('--a_bits', type=int, default=0)
parser.add_argument('--B_bits', type=int, default=0)
parser.add_argument('--n_epoch', type=int, default=10250)
parser.add_argument('--pre_train_n_epoch', type=int, default=300)
parser.add_argument('--save_epoch', type=int, default=10)
parser.add_argument('--save_batch', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_div', type=float, default=0.001)
parser.add_argument('--lr_mul', type=float, default=1000)
parser.add_argument('--ls', type=float, default=0.2)
parser.add_argument('--ld', type=float, default=0)
parser.add_argument('--lssim', type=float, default=0)
parser.add_argument('--lg', type=float, default=0)
parser.add_argument('--l2', type=float, default=1)
parser.add_argument('--l1', type=float, default=0)
parser.add_argument('--device_id' , type=int, default=0)
parser.add_argument('--ol' , type=int, default=0)
parser.add_argument('--th_b' , type=str, default='0')
parser.add_argument('--th_dct' , type=int, default=0)
parser.add_argument('--th_ac'  , type=int, default=0)
parser.add_argument('--ls_start'  , type=int, default=0)
parser.add_argument('--regular'  , type=str, default='l0')
parser.add_argument('--diff'  , type=str, default='')
parser.add_argument('--re_n_epoch', type=int, default=0)

# init setting
parser.add_argument('--init', type=str, default="ae")
parser.add_argument('--K_init', type=int, default=4)
parser.add_argument('--init_batch' , type=int, default=0)
parser.add_argument('--B_init' , type=float, default=1/0.035)

#
parser.add_argument('--u_min' , type=float, default=0)
parser.add_argument('--u_max' , type=float, default=1)
parser.add_argument('--a_min' , type=float, default=0)
parser.add_argument('--a_max' , type=float, default=1)
parser.add_argument('--B_min' , type=float, default=-1000)
parser.add_argument('--B_max' , type=float, default=1000)

parser.add_argument('--beta' , type=float, default=0.5)
parser.add_argument('--gamma' , type=float, default=-0.1)
parser.add_argument('--lam' , type=float, default=1.1)

parser.add_argument('--noise_var' , type=float, default=0.01)
parser.add_argument('--noise_type', type=str, default='speckle')

parser.add_argument('--init_K' , type=int, default=20)
parser.add_argument('--init_regular', type=str, default='no')

parser.add_argument('--target_q', type=int, default=25)

parser.add_argument('--partition', type=str, default="valL")
#   args
args = parser.parse_args()

data_path  = args.data_path
map_path    = args.map_path
#data_path  = '../images/iscas/noisy/noise_type_{}_var_{:.02f}/{}_noise_type_{}_var_{:.02f}.png'.\
#             format(args.noise_type,args.noise_var,args.name,args.noise_type,args.noise_var)
result_path   = args.result_path
init_para_path   = args.init_para_path
init        = args.init
regular     = args.regular
steer  = args.steer
single = args.single
load   = args.load

u_bits      = args.u_bits
a_bits      = args.a_bits
B_bits      = args.B_bits
pre_train_n_epoch = args.pre_train_n_epoch

save_epoch  = args.save_epoch
save_batch  = args.save_batch
K_init = args.K_init
diff   = args.diff

batch_size  = args.batch_size
lr          = args.lr
device_id   = args.device_id
lr_div      = args.lr_div
lr_mul      = args.lr_mul
ls          = args.ls
ld          = args.ld
lssim       = args.lssim
lg          = args.lg
l2          = args.l2
l1          = args.l1
init_batch  = args.init_batch
B_init      = args.B_init
ls_start    = args.ls_start
init_K = args.init_K
init_regular = args.init_regular
target_q = args.target_q

u_min = args.u_min
u_max = args.u_max
a_min = args.a_min
a_max = args.a_max
B_min = args.B_min
B_max = args.B_max

re_n_epoch = args.re_n_epoch

partition = args.partition
if steer:
    single = False

#data_path = './train/images'
#result_path = './results/final_test'
#result_path = './model_weights/final_test'
if not os.path.isdir(result_path):
    os.mkdir(result_path)

#for n in np.arange(50,101,10):
images = []
ori_images = []
c = 0

n_epoch = pre_train_n_epoch + 50*re_n_epoch
seg = "MDBSCAN"
for target_q in [30,25]:
    for init_regular in [args.init_regular]:#,"l0","l1"
        for regular in ["no"]: # "l0","l1"
            for init_K in [args.init_K]: # 1,2,3,4,5,100,150,200,250,300,350,400,450
                for diff in ['40','50']: #'','20','30','40','50','60','70','80','90','100'

                    #map_path = os.path.join(seg.upper(),diff,'{}x{}-seg'.format(bz_seg,bz_seg),partition)
                    data_path = os.path.join('data','img',partition)
                    
                    # create result file
                    if not os.path.join(result_path):
                        os.mkdir(result_path)
                    if not os.path.isdir(os.path.join(args.result_path,seg.lower())):
                        os.mkdir(os.path.join(args.result_path,seg.lower()))
                    if not os.path.isdir(os.path.join(args.result_path,seg.lower(),diff)):
                        os.mkdir(os.path.join(args.result_path,seg.lower(),diff))

                    result_path = os.path.join(args.result_path,seg.lower(),diff)
                    
                    
                    #result_path = os.path.join(args.result_path,seg.upper(),diff)
                    print(data_path,result_path,map_path)
                    

                    for n in sorted(os.listdir(os.path.join(data_path)))[:]:
                        
                        

                        data_name = n[:-4]
                        if data_name == 'parrot':
                            if target_q == 25:
                                continue
                        elif data_name == 'peppers' or data_name == 'flower':
                            data_name = data_name
                        else:
                            continue

                        data = ImageOps.grayscale(Image.open(os.path.join(data_path,n)))
                        print(data.size[0]//4,data.size[1]//4)
                        if data_name == 'baboon':
                            continue
                        #if data_name == 'cameraman':
                        #    data = data
                        #else:
                        #    data = data.resize((data.size[0]//4,data.size[1]//4))
                        data = T.ToTensor()(np.array(data))
                        data = torch.squeeze(data)
                        
                        #print(os.path.join(init_para_path,'steer_{}-single_{}-K_{:03d}-regular_{}'.format(steer,single,init_K,init_regular),'{}.pkl'.format(data_name)))
                        with open(os.path.join(init_para_path,seg.lower(),diff,'steer_{}-single_{}-K_{:03d}-target_{:d}-regular-{}'.format(steer,single,init_K,target_q,init_regular),'{}.pkl'.format(data_name)), 'rb') as fp:
                            init_para = pickle.load(fp)
                        #print('{}.pkl'.format(data_name))
                        #with open('{}.pkl'.format(data_name), 'rb') as fp:
                        #     init_para = pickle.load(fp)


                        u = init_para['u']
                        a = init_para['a']
                        B = init_para['B']
                        if init_regular != 'no':
                            p = init_para['p']

                        K_init = u.shape[1]

                        print(data.shape,data_name,u.shape,a.shape,B.shape,K_init)

                        #bz = data.shape[1]
                        segs = data.unsqueeze(0)
                        n_seg = len(segs)

                        if u_bits == 0 and a_bits == 0 and B_bits == 0:
                            limit = False
                            quant = False
                        else:
                            limit = False
                            quant = True

                        devices = ['cuda:{:d}'.format(i) for i in range(torch.cuda.device_count())]
                        if device_id == -1 or len(devices) == 0:
                            devices.insert(0,'cpu')

                        file_name = 'steer_{}-single_{}-initre_{}-target_{:d}-initK_{:02d}-regular_{}'\
                                .format(steer,single,init_regular,target_q,init_K,regular)
                        print(file_name)
                        

                        if not os.path.isdir(os.path.join(result_path,file_name)):
                            os.mkdir(os.path.join(result_path,file_name))
                        

                        class Model(nn.Module):
                            def __init__(self,K_init,segs,quant,limit,steer,single,init_para):
                                super().__init__()
                                # To make "a" and "b" real parameters of the model, we need to wrap them with nn.Parameter
                                self.K_init  = K_init
                                self.segs = segs.to(devices[0])
                                self.quant = quant
                                self.limit = limit
                                self.steer = steer
                                self.single = single
                                self.init_para = init_para
                                self.n_seg = len(self.segs)
                                self.K = self.n_seg * self.K_init
                                self.W = segs.shape[1]
                                self.H = segs.shape[2]

                                self.u = nn.Parameter(init_para['u']  , requires_grad=True).to(devices[0])
                                self.a = nn.Parameter(init_para['a']  , requires_grad=True).to(devices[0])
                                if regular != 'no':
                                    if len(init_para['p']) > 0:
                                        self.p = nn.Parameter(init_para['p']  , requires_grad=True).to(devices[0])
                                    else: 
                                        self.p = nn.Parameter(torch.tensor(np.ones((self.n_seg,self.K_init))/self.n_seg/self.K_init, requires_grad=True, dtype=torch.float).to(devices[0]))
                                self.B = nn.Parameter(init_para['B'], requires_grad=True).to(devices[0])
                                
                                #self.init_process()

                            def init_process(self):

                                # B init
                                # if self.steer:
                                #     k_dim = int(np.sqrt(self.K_init))
                                #     sig1 = (1/0.035)**(1/2) / 1 # 1 / (2 * (k_dim + 1))
                                #     sig2 = (1/0.035)**(1/2) / 1 #1 / (2 * (k_dim + 1))
                                #     roh = 0.
                                #     B_per_K = np.array([[sig1 * sig1, roh * sig1 * sig2],
                                #                         [roh * sig1 * sig2, sig2 * sig2]])
                                #     #B_per_K = np.linalg.cholesky(np.linalg.inv(B_per_K))
                                # else:
                                #     B_per_K = 1/0.035
                                
                                # init_time_start = datetime.now()
                                
                                # Bi = []
                                
                                # ui,ai = self.random_init()
                                # Bi = torch.tile(torch.tensor(B_per_K),(self.n_seg,self.K_init,1,1))
                                
                                # ui = ui.reshape(self.n_seg,self.K_init,2)
                                # ai = ai.reshape(self.n_seg,self.K_init)
                                # Bi = Bi.reshape(self.n_seg,self.K_init,2,2)


                                # self.u.data = nn.Parameter(ui, requires_grad=True).to(devices[0])
                                # self.a.data = nn.Parameter(ai, requires_grad=True).to(devices[0])
                                # self.B.data = nn.Parameter(Bi, requires_grad=True).to(devices[0])
                                # if regular != 'no':
                                #     self.p = nn.Parameter(torch.tensor(np.ones((self.n_seg,self.K_init))/self.n_seg/self.K_init, requires_grad=True, dtype=torch.float).to(devices[0]))
                                
                                
                                # #self.pis_mask.data = (self.p > 0)
                                # #if self.block:
                                # #    self.B.data = nn.Parameter(block_B).to(devices[0])
                                # #else:
                                # #    self.B.data = nn.Parameter(block_B*(self.data.shape[0]//self.bz_init)).to(devices[0])
                                print('u shape {}'.format(self.u.shape))
                                print('a shape {}'.format(self.a.shape))
                                print('B shape {}'.format(self.B.shape))
                                if regular != 'no':
                                    print('p shape {}'.format(self.p.shape))
                                
                                #init_time = datetime.now() - init_time_start
                                #print('{} init time : {:2f} sec'.format(self.init,init_time.total_seconds()))

                            # def grid_init(self):

                            #     n = int(np.sqrt(self.K_init))
                            #     fn = []
                            #     for x in range(2,n+1):
                            #         if self.K_init%x == 0:
                            #             fn.append(x)
                            #     K_dim1 = fn[-1]
                            #     K_dim2 = self.K_init//K_dim1
                            #     #K_dim = int(np.sqrt(self.K_init))
                            #     x = torch.linspace(0, self.segs.shape[1], K_dim1+1)

                            #     #x = x+(x[0]+x[1])/2
                            #     x = x[:-1]
                            #     y = torch.linspace(0, self.segs.shape[2], K_dim2+1)

                            #     #y = y+(y[0]+y[1])/2
                            #     y = y[:-1]
                            #     #print(x,y)
                            #     xx, yy = torch.meshgrid(x, y)
                            #     location = torch.cat((xx.reshape(1,-1),yy.reshape(1,-1)),0).permute((1,0))/self.bz_init
                            #     locations = torch.tile(location,(len(self.segs),1,1))
                            #     #print('grid location shape {}, locations shape {}'.format(location.shape,locations.shape))
                            #     v = torch.zeros((len(self.segs),self.K)  , dtype=torch.float).to(devices[0])
                            #     for n, block in enumerate(self.segs):
                            #         for k in range(self.K_init):
                            #             [l1,l2] = location[k] * torch.tensor([self.bz_init,self.bz_init])

                            #             #v[n,k] = torch.mean(block[int(l2-y[0]):int(l2+y[0]),int(l1-x[0]):int(l1+x[0])])
                            #             v[n,k] = torch.mean(block[int(l1):int(l1+x[1]),int(l2):int(l2+y[1])])
                            #             #print(block.shape,v[n,k],int(l1-x[0]),int(l1+x[0]),int(l2-y[0]),int(l2+y[0]))
                            #     #print(locations)
                            #     return locations.to(devices[0]),v.to(devices[0])

                            # def kmean_init(self):
                            #     #init_time_start = datetime.now()
                            #     km = KMeans(n_clusters=self.K_init)
                            #     locations = torch.ones((len(self.segs),self.K_init,2), dtype=torch.float).to(devices[0])
                            #     v = torch.ones((len(self.segs),self.K_init), dtype=torch.float).to(devices[0])
                            #     #locations = torch.tensor([])
                            #     for n, block in enumerate(self.segs):
                            #         km.fit(block.cpu().reshape(-1,1))

                            #         for k in range(self.K_init):

                            #             avail_pixels = (np.where(km.labels_.reshape((block.shape)) == k))
                                        
                            #             #print(km.cluster_centers_[k])
                            #             if len(avail_pixels[0]) > 0:
                            #                 selected = np.random.choice(len(avail_pixels[0]), replace=False)
                            #                 locations[n,k] = torch.tensor([avail_pixels[0][selected]/self.bz_init,avail_pixels[1][selected]/self.bz_init])
                            #                 v[n,k] = torch.tensor(km.cluster_centers_[k])
                            #             else:
                            #                 locations[n,k] = torch.tensor([0.5,0.5])
                            #                 v[n,k] = torch.tensor(block[int(0.5*self.bz_init),int(0.5*self.bz_init)])
                            #     #print(locations.shape,v.shape)

                            #     return locations.to(devices[0]),v.to(devices[0])

                            # def random_init(self):
                            #     locations = torch.rand((self.n_seg,self.K_init,2), dtype=torch.float)
                            #     # locations = torch.tensor([])
                            #     # for l in range(label.min(),label.max()+1):
                            #     #     avail_pixels = np.where(label==l)
                            #     #     #print(m)
                            #     #     n_K = np.min([len(avail_pixels[0]),self.K_init])
                            #     #     selected = np.random.choice(len(avail_pixels[0]),n_K, replace=False)
                                    
                            #     #     #locations = torch.tensor((len(blocks_init),self.K_init,2), dtype=torch.float).to(devices[0])
                            #     #     location  = torch.tensor([avail_pixels[0][selected]/self.bz_seg,avail_pixels[1][selected]/self.bz_seg], dtype=torch.float).permute(1,0)
                            #     #     locations = torch.cat((locations,location),0)
                            #     # #locations = locations.unsqueeze(0)
                            #     v = torch.rand((self.n_seg,self.K_init), dtype=torch.float)
                            #     #print(self.segs.shape)
                            #     print(locations.shape)
                            #     for n, block in enumerate(self.segs):
                            #         for k in range(self.K_init):
                            #             [x,y] = locations[n,k] * torch.tensor([self.segs.shape[1],self.segs.shape[2]])
                            #             v[n,k] = block[int(x),int(y)]

                            #     return locations.to(devices[0]),v.to(devices[0])
                            
                            
                            def quantize(self, x, x_min, x_max, bits):
                                ans = ((x-x_min)/(x_max-x_min)*(2**bits-1)).round() * (x_max-x_min)/(2**bits-1) + x_min
                                ans[ans>x_max] = x_max
                                ans[ans<x_min] = x_min
                                return ans

                            def limitation(self, x, x_min, x_max):
                                ans = x.clone()
                                ans[ans > x_max] = x_max
                                ans[ans < x_min] = x_min
                                return ans

                            def clone_kernel(self):
                                a = self.a.data.clone().tile(1,2)
                                u = self.u.data.clone().tile(1,2,1)
                                B = self.B.data.clone().tile(1,2,1,1)
                                if regular != 'no':
                                    p = self.p.data.clone().tile(1,2)
                                with torch.no_grad():
                                    self.u = nn.Parameter(torch.cat([self.u.data,self.u.data+torch.randn((1, self.K_init,2)).to(devices[0])/10],dim=1), requires_grad=True).to(devices[0])
                                    self.a = nn.Parameter(torch.cat([self.a.data,self.a.data],dim=1), requires_grad=True).to(devices[0])
                                    self.B = nn.Parameter(torch.cat([self.B.data,self.B.data],dim=1), requires_grad=True).to(devices[0])
                                    if regular != 'no':
                                        self.p = nn.Parameter(torch.cat([self.p.data,self.p.data],dim=1), requires_grad=True).to(devices[0])
                                self.K_init += self.K_init
                                print('u shape {}'.format(self.u.shape))
                                print('a shape {}'.format(self.a.shape))
                                print('B shape {}'.format(self.B.shape))
                                if regular != 'no':
                                    print('p shape {}'.format(self.p.shape))
                                    
                            def gaussian(self, center, B,bz1 ,bz2,x1,y1):
                                B = B.float()
                                #center = center.cpu()
                                x_axis = torch.linspace(x1, (x1+bz1)-1, bz1).to(devices[0])/self.W - center[0]
                                y_axis = torch.linspace(y1, (y1+bz2)-1, bz2).to(devices[0])/self.H - center[1]
                                xx, yy = torch.meshgrid(x_axis, y_axis)
                                if self.steer:
                                    data1 = torch.cat((torch.unsqueeze(xx.reshape(-1),-1),torch.unsqueeze(yy.reshape(-1),-1)),1)
                                    data1 = torch.unsqueeze(data1,1)
                                    data2 = torch.transpose(data1,1,2)
                                    #print(data1.shape,data2.shape,B.shape)
                                    kernel = torch.matmul(data1,torch.transpose(B,0,1))
                                    kernel = torch.matmul(kernel,B)
                                    kernel = torch.matmul(kernel,data2)
                                    #print(t.shape)
                                    kernel = torch.exp(-0.5 * kernel)
                                    kernel = torch.squeeze(kernel)
                                else:
                                    kernel = torch.exp(-B * (xx**2 + yy**2))
                                #print(kernel.shape)
                                return kernel.reshape(-1)
                                
                            def forward(self,x1,y1,bz1,bz2):
                                
                                k = torch.randn(self.n_seg,self.K_init,bz1*bz2, dtype=torch.float).to(devices[0])
                                #g = torch.randn(self.n_seg,self.K_init,self.W*self.H, dtype=torch.float).to(devices[0])
                                #print('k shape = {}, g shape = {}, batch_size = {}, block index = {}'.format(k.shape,g.shape,batch_size,block_index))
                                for n_s in range(self.n_seg):
                                    for i in range(self.K_init):
                                        if self.single:
                                            k[n_s][i] = self.gaussian(center = self.u[n_s][i], B = self.B, bz1=bz1,bz2=bz2,x1=x1,y1=y1).to(devices[0])
                                        elif self.steer:
                                            k[n_s][i] = self.gaussian(center = self.u[n_s][i], B = torch.tril(self.B[n_s][i]),bz1=bz1,bz2=bz2,x1=x1,y1=y1).to(devices[0])#torch.tril(self.B[n_b][i]))#
                                        else:
                                            k[n_s][i] = self.gaussian(center = self.u[n_s][i], B = self.B[n_s][i])

                                if regular == 'l0':
                                    mask = torch.sigmoid(self.p)*(args.lam-args.gamma)+args.gamma
                                    mask[mask<=0] = torch.tensor(0)
                                    mask[mask>=1] = torch.tensor(1)
                                    k = k * torch.unsqueeze(mask,-1)
                                elif regular == 'l1':
                                    mask = torch.zeros(self.p.shape).to(devices[0])
                                    mask[self.p > 0] = self.p[self.p > 0]
                                    k = k * torch.unsqueeze(mask,-1)
                                else:
                                    k = k
                                    
                            

                                k_sum = torch.sum(k,1)
                                k_sum[k_sum < 10e-8] = 10e-8
                                k_sum = torch.unsqueeze(k_sum, 1)
                                g = torch.div(k,k_sum)

                                z = torch.unsqueeze(self.a,-1)*g
                                z[z < 0] = 0
                                z[z > 1] = 1
                                z = torch.sum(z,1)
                                z = z.reshape(self.n_seg,bz1,bz2)

                                return z

                            def global_re(self):
                                k = torch.randn(1,self.n_seg*self.K_init,128*128, dtype=torch.float).to(devices[0])
                                g = torch.randn(1,self.n_seg*self.K_init,128*128, dtype=torch.float).to(devices[0])
                                a = self.a.reshape(1,-1)#  torch.randn(1,self.n_seg*self.K_init, dtype=torch.float).to(devices[0])
                                B = self.B.reshape(1,-1,2,2)
                                u = self.u.reshape(1,-1,2)
                                p = self.p.reshape(1,-1)
                                #p = torch.randn(1,self.n_seg*self.K_init, dtype=torch.float).to(devices[0])
                                #print('k shape = {}, g shape = {}, batch_size = {}, block index = {}'.format(k.shape,g.shape,batch_size,block_index))
                                for i in range(u.shape[1]):
                                    k[0][i] = self.gaussian(center = u[0][i], B = torch.tril(B[0][i]),bz1=128,bz2=128)#torch.tril(self.B[n_b][i]))#
                                        
                                        # tB[0,0] = tB[0,0] / 128 * shapes[n_s][0]
                                        # tB[1,1] = tB[1,1] / 128 * shapes[n_s][1]
                                        # k[0][n_s*self.K_init+i] = self.gaussian(center = np.array(locs[n_s])/128, B = torch.tril(tB),bz1=128,bz2=128)#torch.tril(self.B[n_b][i]))#
                                        # a[0][n_s*self.K_init+i] = self.a[n_s][i]
                                        # p[0][n_s*self.K_init+i] = self.p[n_s][i]

                                if regular == 'l0':
                                    mask = torch.sigmoid(p)*(args.lam-args.gamma)+args.gamma
                                    mask[mask<=0] = torch.tensor(0)
                                    mask[mask>=1] = torch.tensor(1)
                                    k = k * torch.unsqueeze(mask,-1)
                                elif regular == 'l1':
                                    mask = torch.zeros(p.shape).to(devices[0])
                                    mask[p > 0] = p[p > 0]
                                    k = k * torch.unsqueeze(mask,-1)
                                else:
                                    k = k
                                    
                            

                                k_sum = torch.sum(k,1)
                                k_sum[k_sum < 10e-8] = 10e-8
                                k_sum = torch.unsqueeze(k_sum, 1)
                                g = torch.div(k,k_sum)

                                z = torch.unsqueeze(a,-1)*g
                                z[z < 0] = 0
                                z[z > 1] = 1
                                z = torch.sum(z,1)
                                z = z.reshape(1,128,128)

                                return z
                            
                        def Reconstruction(save_name):
                            if save_name == "None":
                                save_name = os.path.join(result_path,file_name,data_name)

                            if limit:
                                PATH = '{}/{}/{}_limit.pth'.format(result_path,file_name,data_name)
                            elif quant:
                                PATH = '{}/{}/{}_quant.pth'.format(result_path,file_name,data_name)
                            else:
                                PATH = '{}/{}/{}.pth'.format(result_path,file_name,data_name)

                            model_re = copy.deepcopy(model).to(devices[0])

                            print(PATH)
                            if os.path.exists(PATH):
                                print("Load saved model !!!")
                                model_re.load_state_dict(torch.load(PATH))

                            if regular == 'l0':
                                mask = torch.sigmoid(model_re.p)*(args.lam-args.gamma)+args.gamma
                            elif regular == 'l1':
                                mask = torch.zeros(model_re.p.shape).to(devices[0])
                                mask[model_re.p > 0] = model_re.p[model_re.p > 0]
                            else:
                                mask = torch.ones(model_re.a.shape)
                            
                            print('B max : {} B min : {}\n\
                                u max : {} u min : {}\n\
                                a max : {} a min : {}\n\
                                mask max : {} mask min : {}\n\
                                original number of kernels: {}\n\
                                number of activated kernel: {}'\
                                .format(model_re.B.max(),model_re.B.min(),model_re.u.max(),model_re.u.min(),\
                                        model_re.a.max(),model_re.a.min(),mask.max(),mask.min(),model_re.K_init,(mask > 0).sum()))

                            #print(R,C,n_block)
                            image_recon = np.zeros(data.shape)
                            for x1 in np.arange(0,data.shape[0],data.shape[0]//2):
                                for y1 in np.arange(0,data.shape[1],data.shape[1]//2):
                            
                                    #print(x1,y1,batch_gt[:,x1:x1+data.shape[0]//4,y1:y1+data.shape[1]//2].shape)
                                    print(x1,y1)
                                    with torch.no_grad():
                                        batch_yp = model(x1,y1,data.shape[0]//2,data.shape[1]//2)
                                    image_recon[x1:x1+data.shape[0]//2,y1:y1+data.shape[1]//2] = batch_yp.squeeze(0).cpu().detach().numpy()
                            del model_re
                            #image_recon = torch.zeros(data.shape, dtype=torch.float).to(devices[0])
                            #n_model     = torch.zeros(data.shape, dtype=torch.float).to(devices[0])
                            #c = 0
                            
                            #image_recon = yp.squeeze(0).cpu().detach().numpy()
                            #print(image_recon.max(),image_recon.min())
                            #orig_image = plt.imread('../images/32x32/img_barb_crop_2.png')#.cpu().detach().numpy()
                            #bm3d_recon = np.array(Image.open(os.path.join('../images/iscas/bm3d/noise_type_{}_var_{:.02f}'.format(args.noise_type,args.noise_var),'{}.png'.format(data_name[:15]))))
                            #bm3d_recon = bm3d_recon.squeeze()/255.
                            #orig_image = np.array(Image.open(os.path.join('../images/iscas/noisy_free/','{}.png'.format(data_name[:15]))))
                            #orig_image = orig_image.squeeze()/255.
                            print(data.shape)
                            orig_image = data[:data.shape[0],:data.shape[1]].squeeze(0).cpu().detach().numpy()
                            print(image_recon.shape,orig_image.shape)
                            # save reconstructed and compare images
                            #plt.imsave('{}.jpg'.format(save_name),image_recon,cmap='gray',vmin=0,vmax=1)
                            plt.imsave('{}.jpg'.format(save_name),image_recon,cmap='gray',vmax = 1,vmin = 0)
                            #print(image_recon.max(),bm3d_recon.max(),orig_image.max(),noisy_image.max())

                            psnr = peak_signal_noise_ratio((orig_image*255).astype(np.uint8),(image_recon*255).astype(np.uint8),data_range = 255)
                            ssim = structural_similarity((orig_image*255).astype(np.uint8),(image_recon*255).astype(np.uint8),data_range = 255)
                            #psnr_bm = peak_signal_noise_ratio((orig_image*255).astype(np.uint8),(bm3d_recon*255).astype(np.uint8),data_range = 255)
                            #ssim_bm = structural_similarity((orig_image*255).astype(np.uint8),(bm3d_recon*255).astype(np.uint8),data_range = 255)

                            f,ax = plt.subplots(1,2,figsize=(18,9))
                            f.suptitle('Data name: {}'.format(data_name), fontsize=20)
                            ax[0].imshow(image_recon,'gray',vmax = 1,vmin = 0)
                            ax[0].set_title('Init: {} diff: {}\nPSNR: {:.2f}\nSSIM: {:.2f}\n\
                                            original number of kernels: {:3d}\nnumber of kernels: {:3d}'.\
                                            format(seg,diff,psnr,ssim,model.K_init,(mask>0).sum()), fontsize=15)
                            ax[1].imshow(orig_image,'gray',vmax = 1,vmin = 0)
                            ax[1].set_title('orig_image', fontsize=15)
                            f.savefig('{}_compare.jpg'.format(save_name))
                            plt.close(f)

                            # Calculate PSNR and SSIM

                            #print('training time : {:.4f}, iteration : {:.2f}'.format(np.sum(times)/len(times), np.sum(iters)/len(iters)))
                        # end reconstruction

                        def save_pretrain(index,model):
                            pretrainU = model.u[index].data
                            pretrainA = model.a[index].data
                            if not single:
                                pretrainB = model.B[index].data
                            else:
                                pretrainB = model.B.data
                            if regular != 'no':
                                pretrainP = model.p[index].data
                                return [pretrainU,pretrainA,pretrainB,pretrainP]
                            return [pretrainU,pretrainA,pretrainB]

                        def detach_pretrain(index,model):
                            model.u[index].detach()
                            model.a[index].detach()
                            if not single:
                                model.B[index].detach()
                            if regular != 'no':
                                model.p[index].detach()

                        def retain_pretrain(index,model,P):
                            model.u[index].data = P[0]
                            model.a[index].data = P[1]
                            if not single:
                                model.B[index].data = P[2]
                            if regular != 'no':
                                model.p[index].data = P[3]
                        # Start to train !!!================================================================================================

                        torch.manual_seed(7)
                        loss_fn  = nn.MSELoss()

                        # initialization
                        #current_loss = 1e6
                        #loss = 0
                        #f_current_loss = 1e6
                        #f_loss = 0

                        # prepare model
                        model = Model(K_init,segs,quant,limit,steer,single,init_para).to(devices[0])
                        model.train()
                        
                        # prepare ground truth
                        gt = model.segs
                        print('gt shape = {}'.format(gt.shape))
                        # R = int(data.shape[0]/bz)
                        # C = int(data.shape[1]/bz)
                        # n_block = R * C
                        # gt = torch.zeros((n_block,bz+ol,bz+ol), dtype=torch.float).to(devices[0])
                        # for n in range(R):
                        #     for m in range(C):
                        #         gt[C*n+m] = data[bz*n:bz*(n+1)+ol,bz*m:bz*(m+1)+ol].to(devices[0])




                        #optimizer = optim.Adam(model.parameters(), lr=lr)
                        #optimizer = optim.Adam([model.a, model.u], lr=lr)
                        #optimizer_p = optim.Adam([model.p], lr=lr/lr_div)
                        #optimizer_B = optim.Adam([model.B], lr=lr*lr_mul)
                        # load model weights
                        # if init.lower() == "block":
                        #     model.block_init(init_path)
                        # elif init.lower() == "bass":
                        #     model.BASS_init(init_path)
                        # elif init.lower() == 'grid':
                        #     model.grid_init()
                        # elif init.lower() == 'ae':
                        #     model.AE_init(init_path)
                        # elif init.lower() == 'kmean':
                        #     model.kmean_init()
                        # else:
                        #     model.random_init()


                        if model.u.shape[0] < batch_size:
                            batch_size = model.u.shape[0]
                        n_batch = model.u.shape[0] // batch_size
                        print(n_batch)

                        #limit = False
                        #quant = False
                        #load = True
                        if not load:
                            if limit:
                                PATH = '{}/{}/{}_limit.pth'.format(result_path,file_name,data_name)
                            elif quant:
                                PATH = '{}/{}/{}_quant.pth'.format(result_path,file_name,data_name)
                            else:
                                PATH = '{}/{}/{}.pth'.format(result_path,file_name,data_name)
                            torch.save(model.state_dict(), PATH)
                            print(model.u.shape,model.B.shape,model.a.shape)
                            save_name = '{}/{}/{}_init'.format(result_path,file_name,data_name)
                            #Reconstruction(save_name)
                        else:
                            if limit:
                                PATH = '{}/{}/{}_limit.pth'.format(result_path,file_name,data_name)
                            elif quant:
                                PATH = '{}/{}/{}_quant.pth'.format(result_path,file_name,data_name)
                            else:
                                PATH = '{}/{}/{}.pth'.format(result_path,file_name,data_name)
                            #PATH = '{}/{}/{}.pth'.format(result_path,file_name,data_name)
                            if os.path.exists(PATH):
                                model.load_state_dict(torch.load(PATH))
                                save_name = '{}/{}/{}_load'.format(result_path,file_name,data_name)
                                #Reconstruction(save_name)
                                init_batch = n_batch + 1


                        #save_name = '{}/{}/{}_init'.format(result_path,file_name,data_name)
                        #print("save name [-4:] ", save_name[-4:])
                        #Reconstruction(save_name)
                        ls_interval = np.linspace(0.1,15,50)

                        #permutation = torch.arange(n_block)

                        time_start = datetime.now()
                        losses   = np.zeros((n_batch,n_epoch))
                        #f_losses = np.zeros((n_batch,n_epoch))
                        times    = np.zeros((n_batch,n_epoch))
                        #iters    = np.zeros(n_batch,n_epoch)
                        #print(' n block:{}\n n batch:{}\n batch size:{}\n block size:{}'.format(n_block,n_batch,batch_size,bz))
                        for batch in range(init_batch,n_batch):

                            #print(batch)
                            batch_start = datetime.now()
                            block_index = torch.arange(batch*batch_size,(batch+1)*batch_size)
                            batch_gt = gt[block_index]

                            m = torch.zeros(model.u.shape[0])
                            m[block_index] = 1
                            P = save_pretrain(torch.arange(model.u.shape[0])[m==0],model)
                            detach_pretrain(torch.arange(model.u.shape[0])[m==0],model)

                            model.train()
                            #optimizer = optim.Adam(model.parameters(), lr=lr)
                            optimizer   = optim.Adam([model.a, model.u], lr=lr)
                            if regular != 'no':
                                optimizer_p = optim.Adam([model.p], lr=lr/lr_div)
                            optimizer_B = optim.Adam([model.B], lr=lr*lr_mul)

                            current_loss = 1e6
                            loss = 0
                            #f_current_loss = 1e6
                            #f_loss = 0

                            loss_s = 0
                            loss_d = 0
                            loss_g = 0
                            loss_ssim = 0
                            loss_mse  = 0

                            for epoch in range(n_epoch):
                                #print(epoch)
                                epoch_start = datetime.now()
                                # #ls = ls_interval[int(epoch/50)]/K/K
                                if epoch > pre_train_n_epoch:
                                    ls = ls_interval[(epoch-pre_train_n_epoch)//re_n_epoch]**2/model.K
                                    #ld = 2*(10**-8)
                                else:
                                    ls = 0

                                
                                
                                loss = 0
                                r1 = 4
                                r2 = 4
                                for x1 in np.arange(0,data.shape[0],data.shape[0]//r1):
                                    for y1 in np.arange(0,data.shape[1],data.shape[1]//r2):
                                    
                                        optimizer.zero_grad()
                                        if regular != 'no':
                                            optimizer_p.zero_grad()
                                        optimizer_B.zero_grad()
                                
                                        print(x1,y1,batch_gt[:,x1:x1+data.shape[0]//r1,y1:y1+data.shape[1]//r2].shape)
                                        batch_yp = model(x1,y1,data.shape[0]//r1,data.shape[1]//r2)
                                        loss_l2 = ( (batch_yp.to(devices[0])-batch_gt[:,x1:x1+data.shape[0]//r1,y1:y1+data.shape[1]//r2])**2*1000).mean().to(devices[0])
                                        loss_l1 = 0
                                        #loss_l1  = ( torch.abs(batch_yp.to(devices[0])-batch_gt) ).mean().to(devices[0])
                                
                                        if regular == 'l0':
                                            #loss_s = (model.p>1/model.n_block/model.K*th_ac).sum().to(devices[0])
                                            loss_s = torch.sigmoid(model.p-args.beta*np.log(-args.gamma/args.lam)).sum().to(devices[0])
                                        elif regular == 'l1':
                                            loss_s = (model.p[model.p>0]).sum().to(devices[0])
                                        else:
                                            loss_s = 0
                                
                                        if ld > 0:
                                            loss_d = (model.B[:,:,0,0]**2 * model.B[:,:,1,1]**2).mean().to(devices[0])

                                        #print(ls, loss_s)
                                        tloss = loss_l2*l2+loss_s*ls+loss_d*ld+loss_ssim*lssim+loss_g*lg+loss_l1*l1
                                        
                                        
                                        tloss.to(devices[0]).backward()
                                        optimizer.step()
                                        if regular != 'no':
                                            optimizer_p.step()
                                        optimizer_B.step()
                                        
                                        loss += tloss.cpu().detach()
                                        print((datetime.now()-epoch_start).total_seconds())
                                
                                #print(batch_gt.shape,batch_yp.shape)
                                # get loss_ssim
                                if lssim > 0:
                                    #yp = model(torch.arange(n_block)).to(devices[0])
                                    image_recon = torch.zeros(data.shape, dtype=torch.float).to(devices[0])
                                    image_orig  = torch.zeros(data.shape, dtype=torch.float).to(devices[0])
                                    n_model = torch.zeros(data.shape, dtype=torch.float).to(devices[0])
                                    for n in range(R):
                                        for m in range(C):
                                            index = C*n + m
                                            if index in block_index:
                                                image_recon[(bz-ol)*n:(bz-ol)*n+bz,\
                                                            (bz-ol)*m:(bz-ol)*m+bz] += batch_yp[index]
                                                image_orig[(bz-ol)*n:(bz-ol)*n+bz,\
                                                        (bz-ol)*m:(bz-ol)*m+bz] += gt[index]
                                                n_model[(bz-ol)*n:(bz-ol)*n+bz,\
                                                        (bz-ol)*m:(bz-ol)*m+bz] += 1
                                    image_recon = image_recon / n_model
                                    image_orig  = image_orig  / n_model
                                    image_recon[n_model == 0] = 0
                                    image_orig[n_model == 0] = 0
                                            # if n!=0:
                                            #     image_recon[bz*n:bz*n+ol,bz*m:bz*(m+1)+ol] /= 2
                                            # if m!=0:
                                            #     image_recon[bz*n:bz*(n+1)+ol,bz*m:bz*m+ol] /= 2
                                            # if n!=0 and m !=0:
                                            #     image_recon[bz*n:bz*n+ol,bz*m:bz*m+ol] *= 2
                                    loss_ssim = 1- pytorch_ssim.ssim(image_recon.unsqueeze(0).unsqueeze(0),\
                                                                    image_orig.unsqueeze(0).unsqueeze(0))
                                    #loss_ssim = 1- pytorch_ssim.ssim(batch_yp.reshape((-1,1,bz,bz)),\
                                    #                                 batch_gt.reshape((-1,1,bz,bz)))
                                    #ssim = SSIM().to(devices[0])
                                    #loss_ssim = 1-ssim(torch.tile(data[:data.shape[0]-ol,:data.shape[1]-ol].to(devices[0])/data.max(),(1,3,1,1)),torch.tile(image_recon[:data.shape[0]-ol,:data.shape[1]-ol]/image_recon.max(),(1,3,1,1)))
                                    #loss_ssim = 1-structural_similarity(data[:data.shape[0]-ol,:data.shape[1]-ol].cpu().detach().numpy(),image_recon[:data.shape[0]-ol,:data.shape[1]-ol].cpu().detach().numpy(),data_range=1.)
                                # get loss_ssim
                                if lg > 0:
                                    yp = model.global_re()
                                    image_recon = yp
                                    loss_g = ((data.to(devices[0]) - image_recon)**2).mean().to(devices[0])

                                #ma = np.load(os.path.join(map_path,'train_map_{}.npy'.format(data_name[-3:])))
                                #ma = torch.tensor(ma).unsqueeze(0)
                                #avail_pixels = np.where(m==1)
                                
                                #loss_l2 = ( (batch_yp.to(devices[0])-batch_gt)**2*1000).mean().to(devices[0])
                                #loss_l1 = 0
                                ##loss_l1  = ( torch.abs(batch_yp.to(devices[0])-batch_gt) ).mean().to(devices[0])
                                
                                #if regular == 'l0':
                                #    #loss_s = (model.p>1/model.n_block/model.K*th_ac).sum().to(devices[0])
                                #    loss_s = torch.sigmoid(model.p-args.beta*np.log(-args.gamma/args.lam)).sum().to(devices[0])
                                #elif regular == 'l1':
                                #    loss_s = (model.p[model.p>0]).sum().to(devices[0])
                                #else:
                                #    loss_s = 0
                                
                                #if ld > 0:
                                #    loss_d = (model.B[:,:,0,0]**2 * model.B[:,:,1,1]**2).mean().to(devices[0])

                                ##print(ls, loss_s)
                                #loss = loss_l2*l2+loss_s*ls+loss_d*ld+loss_ssim*lssim+loss_g*lg+loss_l1*l1
                                ##print('loss = {:.2f}'.format(loss))
                                if abs(loss - current_loss) < 1e-10 or torch.isnan(loss):
                                    print('terminate !! file name : {} data name : {} epoch: {:4d} loss : {:.3f} rec loss : {:.3f} reg ls : {:.3f}'.\
                                            format(file_name,data_name,epoch,loss,loss_l2*l2+loss_l1*l1,ls*loss_s))
                                    break

                                # save best model weights without limit or quant
                                if loss-loss_s*ls < current_loss and not torch.isnan(loss):
                                    PATH = '{}/{}/{}.pth'.format(result_path,file_name,data_name)
                                    retain_pretrain(torch.arange(model.u.shape[0])[m==0],model,P)
                                    torch.save(model.state_dict(), PATH)
                                    detach_pretrain(torch.arange(model.u.shape[0])[m==0],model)
                                    current_loss = loss-loss_s*ls
                                    #save_name = '{}/{}/{}_updated'.format(result_path,file_name,data_name)
                                    #Reconstruction(save_name)
                                if epoch%100 == 0 and epoch//100 > 0:
                                    PATH = '{}/{}/{}_{}.pth'.format(result_path,file_name,data_name,epoch)
                                    retain_pretrain(torch.arange(model.u.shape[0])[m==0],model,P)
                                    torch.save(model.state_dict(), PATH)
                                    detach_pretrain(torch.arange(model.u.shape[0])[m==0],model)
                                    current_loss = loss-loss_s*ls

                                
                                

                                losses[batch,epoch]   =   loss.cpu().detach().numpy()
                                #f_losses[batch,epoch] = f_loss.cpu().detach().numpy()
                                times[batch,epoch]     = (datetime.now()-epoch_start).total_seconds()
                                # save loss
                                np.save('{}/{}/{}_losses.npy'.format(result_path,file_name,data_name),losses)
                                #np.save('{}/{}/{}_f_losses.npy'.format(result_path,file_name,data_name),f_losses)
                                np.save('{}/{}/{}_time.npy'.format(result_path,file_name,data_name),times)
                                
                                
                                if ((epoch+1) % save_epoch) == 0 and (batch % save_batch) == 0:
                                    save_name = os.path.join(result_path,file_name,'{}_batch_{:04d}_epoch_{:04d}'.format(data_name,batch,epoch))
                                    #save_name = '{}/{}/{}_batch_{:04d}_epoch_{:04d}'.format(result_path,file_name,data_name,batch,epoch)
                                    Reconstruction(save_name)
                                    f = plt.figure()
                                    plt.plot(losses.mean(0),'r',label='loss')
                                    #plt.plot(f_losses.mean(0),'b',label='fake quantization loss')
                                    plt.title('Loss')
                                    plt.xlabel('epoch')
                                    plt.ylabel('loss')
                                    plt.legend()
                                    f.savefig('{}/{}/{}_losses.jpg'.format(result_path,file_name,data_name))
                                    plt.close(f)
                                
                                # if ( (epoch+1) % save_epoch) == 0 and epoch != (n_epoch-1):
                                #     print('copy!!!')
                                #     psnr = 20 * torch.log10(1.0 / torch.sqrt(loss_l2/1000)) 
                                #     if psnr < 25:
                                #         model.clone_kernel()
                                #         print(model.a.shape)
                                #         optimizer   = optim.Adam([model.a, model.u], lr=lr)
                                #         if regular != 'no':
                                #             optimizer_p = optim.Adam([model.p], lr=lr/lr_div)
                                #         optimizer_B = optim.Adam([model.B], lr=lr*lr_mul)
                                # show result
                                print('diff: {} initre: {} target: {} initK: {} epoch: {:4d}/{:4d} time: {:.2f} min {:.2f} sec loss: {:.6f} rec loss : {:.6f} reg ls : {:.6f}'.\
                                        format(diff,init_regular,target_q,init_K,epoch,n_epoch,\
                                            (datetime.now()-epoch_start).total_seconds()//60,\
                                            (datetime.now()-epoch_start).total_seconds()%60,\
                                            loss,loss_l2*l2+loss_l1*l1,ls*loss_s))

                            #model = Model(bz_seg,K_init,data,segs,quant,limit,steer,single,init).to(devices[0])
                            # if init.lower() == "block":
                            #     model.block_init(init_path)
                            # elif init.lower() == "bass":
                            #     model.BASS_init(init_path)
                            # elif init.lower() == 'grid':
                            #     model.grid_init()
                            # elif init.lower() == 'ae':
                            #     model.AE_init(init_path)
                            # elif init.lower() == 'kmean':
                            #     model.kmean_init()
                            # else:
                            #     model.random_init()

                            # if limit:
                            #     PATH = '{}/{}/{}_limit.pth'.format(result_path,file_name,data_name)
                            # elif quant:
                            #     PATH = '{}/{}/{}_quant.pth'.format(result_path,file_name,data_name)
                            # else:
                            #     PATH = '{}/{}/{}.pth'.format(result_path,file_name,data_name)
                            # model.load_state_dict(torch.load(PATH))

                            print('training time for {:04d} / {:04d} batch : {:.2f} min {:2f} sec'.format(batch,n_batch,(datetime.now()-batch_start).total_seconds()//60,(datetime.now()-batch_start).total_seconds()%60))

                        print('model train finish')

                        #save_name = os.path.join(result_path,file_name,'{}_batch_{:04d}_epoch_{:04d}'.format(data_name,batch,epoch))
                        #Reconstruction(save_name)
                        f = plt.figure()
                        plt.plot(losses.mean(0),'r',label='loss')
                        #plt.plot(f_losses.mean(0),'b',label='fake quantization loss')
                        plt.title('Loss')
                        plt.xlabel('epoch')
                        plt.ylabel('loss')
                        plt.legend()
                        f.savefig('{}/{}/{}_losses.jpg'.format(result_path,file_name,data_name))
                        plt.close(f)

                        Reconstruction("None")
                        limit = False
                        quant = False
                        save_name = '{}/{}/{}_Best'.format(result_path,file_name,data_name)
                        Reconstruction(save_name)

