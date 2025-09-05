# %%
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

# %%
parser = argparse.ArgumentParser()
#parser.add_argument('--name', type=str, default="img_lena_crop_2")
parser.add_argument('--init', type=str, default="mdbscan")
parser.add_argument('--diff', type=str, default="20")
parser.add_argument('--regular', type=str, default="regular")
parser.add_argument('--partition', type=str, default="val")
parser.add_argument('--bz_seg', type=int, default=32)
parser.add_argument('--K_init', type=int, default=20)

parser.add_argument('--beta' , type=float, default=0.5)
parser.add_argument('--gamma' , type=float, default=-0.1)
parser.add_argument('--lam' , type=float, default=1.1)

parser.add_argument('--steer' , default=False,action='store_true')
parser.add_argument('--single', default=False,action='store_true')
parser.add_argument('--seg_result_path', type=str, default="train_seg_with_weight")
parser.add_argument('--init_para_path', type=str, default="init_para")
parser.add_argument('--target_q', type=int, default=25)

args = parser.parse_args()

beta = args.beta
gamma = args.gamma
lam = args.lam

init = args.init
diff = args.diff
regular = args.regular
bz_seg = args.bz_seg
K_init = args.K_init
partition = args.partition
steer = args.steer
single = args.single
seg_result_path = args.seg_result_path
init_para_path = args.init_para_path
target_q = args.target_q


# %%
class Model(nn.Module):
    def __init__(self,bz_seg,K_init,data,segs,steer,single):
        super().__init__()
        # To make "a" and "b" real parameters of the model, we need to wrap them with nn.Parameter
        self.bz_seg = bz_seg
        self.K_init  = K_init
        self.data = data.to(devices[0])
        self.segs = segs.to(devices[0])
        self.steer = steer
        self.single = single
        self.n_seg = len(self.segs)
        self.K = self.n_seg * self.K_init
        
        self.u = nn.Parameter(torch.zeros((self.n_seg,self.K_init,2)  , requires_grad=True, dtype=torch.float).to(devices[0]))
        self.a = nn.Parameter(torch.zeros((self.n_seg,self.K_init)    , requires_grad=True, dtype=torch.float).to(devices[0]))
        if regular != 'no':
            self.p = nn.Parameter(torch.tensor(np.ones((self.n_seg,self.K_init))/self.n_seg/self.K_init, requires_grad=True, dtype=torch.float).to(devices[0]))
        self.B = nn.Parameter(torch.tensor(np.ones((self.n_seg,self.K_init,2,2)), dtype=torch.float).to(devices[0]), requires_grad=True)
        self.init_process()
        

    def gaussian(self, center, B,bz1 ,bz2):
        B = B.float()
        x_axis = torch.linspace(0+1/bz1/2, 1-1/bz1/2, bz1).to(devices[0]) - center[0]
        y_axis = torch.linspace(0+1/bz2/2, 1-1/bz2/2, bz2).to(devices[0]) - center[1]
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
        return kernel.reshape(-1)

    def init_process(self):

        # B init
        if self.steer:
            k_dim = int(np.sqrt(self.K_init))
            sig1 = (1/0.035)**(1/2) / 1 # 1 / (2 * (k_dim + 1))
            sig2 = (1/0.035)**(1/2) / 1 #1 / (2 * (k_dim + 1))
            roh = 0.
            B_per_K = np.array([[sig1 * sig1, roh * sig1 * sig2],
                                [roh * sig1 * sig2, sig2 * sig2]])
            #B_per_K = np.linalg.cholesky(np.linalg.inv(B_per_K))
        else:
            B_per_K = 1/0.035
        
        init_time_start = datetime.now()
        
        Bi = []
        ui,ai = self.random_init()
        Bi = torch.tile(torch.tensor(B_per_K),(self.n_seg,self.K_init,1,1))
        
        ui = ui.reshape(self.n_seg,self.K_init,2)
        ai = ai.reshape(self.n_seg,self.K_init)
        Bi = Bi.reshape(self.n_seg,self.K_init,2,2)


        self.u.data = nn.Parameter(ui, requires_grad=True).to(devices[0])
        self.a.data = nn.Parameter(ai, requires_grad=True).to(devices[0])
        self.B.data = nn.Parameter(Bi, requires_grad=True).to(devices[0])
        if regular != 'no':
            self.p = nn.Parameter(torch.tensor(np.ones((self.n_seg,self.K_init))/self.n_seg/self.K_init, requires_grad=True, dtype=torch.float).to(devices[0]))
        
        
        # print('u shape {}'.format(self.u.shape))
        # print('a shape {}'.format(self.a.shape))
        # print('B shape {}'.format(self.B.shape))
        # if regular != 'no':
        #     print('p shape {}'.format(self.p.shape))
        
        # init_time = datetime.now() - init_time_start
        # print('init time : {:2f} sec'.format(init_time.total_seconds()))

    def grid_init(self):

        n = int(np.sqrt(self.K_init))
        fn = []
        for x in range(2,n+1):
            if self.K_init%x == 0:
                fn.append(x)
        K_dim1 = fn[-1]
        K_dim2 = self.K_init//K_dim1
        x = torch.linspace(0, self.segs.shape[1], K_dim1+1)
        x = x[:-1]
        y = torch.linspace(0, self.segs.shape[2], K_dim2+1)
        y = y[:-1]
        xx, yy = torch.meshgrid(x, y)
        location = torch.cat((xx.reshape(1,-1),yy.reshape(1,-1)),0).permute((1,0))/self.bz_init
        locations = torch.tile(location,(len(self.segs),1,1))
        v = torch.zeros((len(self.segs),self.K)  , dtype=torch.float).to(devices[0])
        for n, block in enumerate(self.segs):
            for k in range(self.K_init):
                [l1,l2] = location[k] * torch.tensor([self.bz_init,self.bz_init])
                v[n,k] = torch.mean(block[int(l1):int(l1+x[1]),int(l2):int(l2+y[1])])
        return locations.to(devices[0]),v.to(devices[0])

    def kmean_init(self):
        #init_time_start = datetime.now()
        km = KMeans(n_clusters=self.K_init)
        locations = torch.ones((len(self.segs),self.K_init,2), dtype=torch.float).to(devices[0])
        v = torch.ones((len(self.segs),self.K_init), dtype=torch.float).to(devices[0])
        #locations = torch.tensor([])
        for n, block in enumerate(self.segs):
            km.fit(block.cpu().reshape(-1,1))

            for k in range(self.K_init):

                avail_pixels = (np.where(km.labels_.reshape((block.shape)) == k))
                
                #print(km.cluster_centers_[k])
                if len(avail_pixels[0]) > 0:
                    selected = np.random.choice(len(avail_pixels[0]), replace=False)
                    locations[n,k] = torch.tensor([avail_pixels[0][selected]/self.bz_init,avail_pixels[1][selected]/self.bz_init])
                    v[n,k] = torch.tensor(km.cluster_centers_[k])
                else:
                    locations[n,k] = torch.tensor([0.5,0.5])
                    v[n,k] = torch.tensor(block[int(0.5*self.bz_init),int(0.5*self.bz_init)])

        return locations.to(devices[0]),v.to(devices[0])

    def random_init(self):
        locations = torch.rand((self.n_seg,self.K_init,2), dtype=torch.float)
        v = torch.rand((self.n_seg,self.K_init), dtype=torch.float)
        #print(self.segs.shape)
        #print(locations.shape)
        for n, block in enumerate(self.segs):
            for k in range(self.K_init):
                [x,y] = locations[n,k] * torch.tensor([self.segs.shape[1],self.segs.shape[2]])
                v[n,k] = block[int(x),int(y)]

        return locations.to(devices[0]),v.to(devices[0])
    
    
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
        #print('u shape {}'.format(self.u.shape))
        #print('a shape {}'.format(self.a.shape))
        #print('B shape {}'.format(self.B.shape))
        #if regular != 'no':
        #    print('p shape {}'.format(self.p.shape))

    def forward(self):
        
        k = torch.randn(self.n_seg,self.K_init,self.bz_seg*self.bz_seg, dtype=torch.float).to(devices[0])
        g = torch.randn(self.n_seg,self.K_init,self.bz_seg*self.bz_seg, dtype=torch.float).to(devices[0])
        #print('k shape = {}, g shape = {}, batch_size = {}, block index = {}'.format(k.shape,g.shape,batch_size,block_index))
        for n_s in range(self.n_seg):
            for i in range(self.K_init):
                if self.single:
                    k[n_s][i] = self.gaussian(center = self.u[n_s][i], B = self.B, bz1=self.bz_seg,bz2=self.bz_seg)
                elif self.steer:
                    k[n_s][i] = self.gaussian(center = self.u[n_s][i], B = torch.tril(self.B[n_s][i]),bz1=self.bz_seg,bz2=self.bz_seg)#torch.tril(self.B[n_b][i]))#
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
        z = z.reshape(self.n_seg,self.bz_seg,self.bz_seg)

        return z

    def global_re(self):
        k = torch.randn(1,self.n_seg*self.K_init,128*128, dtype=torch.float).to(devices[0])
        g = torch.randn(1,self.n_seg*self.K_init,128*128, dtype=torch.float).to(devices[0])
        a = self.a.reshape(1,-1)#  torch.randn(1,self.n_seg*self.K_init, dtype=torch.float).to(devices[0])
        B = self.B.reshape(1,-1,2,2)
        u = self.u.reshape(1,-1,2)
        p = self.p.reshape(1,-1)
        for i in range(u.shape[1]):
            k[0][i] = self.gaussian(center = u[0][i], B = torch.tril(B[0][i]),bz1=128,bz2=128)#torch.tril(self.B[n_b][i]))#

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

#%%
devices = ['cuda','cpu']
data_path = os.path.join(init.lower(),diff,'{}x{}-img'.format(bz_seg,bz_seg),partition)
map_path = os.path.join(init.lower(),diff,'{}x{}-seg'.format(bz_seg,bz_seg),partition)
file_name = 'steer_{}-single_{}-K_{:03d}-target_{:d}-regular-{}'.format(steer,single,K_init,target_q,regular)
#result_path = os.path.join('train_nue231',seg.upper(),diff)
seg_result_path = os.path.join(args.seg_result_path,init.lower(),diff)

# %%
if not os.path.isdir(init_para_path):
    os.makedirs(init_para_path)
if not os.path.isdir(os.path.join(init_para_path,init.lower())):
    os.makedirs(os.path.join(init_para_path,init.lower()))
if not os.path.isdir(os.path.join(init_para_path,init.lower(),diff)):
    os.makedirs(os.path.join(init_para_path,init.lower(),diff))
if not os.path.isdir(os.path.join(init_para_path,init.lower(),diff,file_name)):
    os.makedirs(os.path.join(init_para_path,init.lower(),diff,file_name))

save_path = os.path.join(init_para_path,init.lower(),diff,file_name)

print(data_path,map_path,seg_result_path,file_name)
# %%
names = []
maps = []
uss = []
ass = []
Bss = []
pss = []
kss = []
zss = []
times = []
for n in sorted(os.listdir(data_path)[:]):
    
    data_name = n[:-4]
    data = ImageOps.grayscale(Image.open(os.path.join(data_path,n)))
    data = T.ToTensor()(np.array(data))
    data = torch.squeeze(data)
    segmap = np.array(ImageOps.grayscale(Image.open(os.path.join(map_path,n))))
    #print(data.shape,data_name)

    #bz = data.shape[1]
    segs = data.unsqueeze(0)
    model = Model(bz_seg,K_init,data,segs,steer,single)
    PATH = os.path.join(seg_result_path,file_name,'{}.pth'.format(data_name))
    #print(PATH)
    #model.load_state_dict(torch.load(PATH))
    ff = False
    for i in range(3):
        try:
            model.load_state_dict(torch.load(PATH))
            print(n,'  load model!! ', model.K_init)
            K_new = model.K_init
            #print(model.K_init)
            kss.append(K_new)
            
            time = np.load(os.path.join(seg_result_path,file_name,'{}_time.npy'.format(data_name)))
            times.append(time)
            #print(time)
            names.append(data_name)
            #print(data_name)
            maps.append(segmap)
            #print(segmap)
            uss.append(model.u.data)
            ass.append(model.a.data)
            Bss.append(model.B.data)
            pss.append(model.p.data)
            #print(model.u.data)
            
            zss.append(model()[0])
            #print(model()[0])
            
            ff = True
        
        except:
            model.clone_kernel()
        if ff:
            break


# %%
import csv
x1s = []
y1s = []
x2s = []
y2s = [] 
osizes = []
ns = []
for i,n in enumerate(names):
    with open('ori_info_{}_{}_{}.csv'.format(init,diff,partition), 'r') as f:
        csvreader = csv.reader(f)
        for l in csvreader:
            if len(l) > 0 and l[0] == n:
                #print(l[0])
                x1s.append(int(l[1]))
                y1s.append(int(l[2]))
                x2s.append(int(l[3]))
                y2s.append(int(l[4]))
                osizes.append(int(l[5]))

# %%
len(uss),len(osizes)

# %%
def gaussian_B(center, B,bz1 ,bz2):
    B = B.float()
    x_axis = torch.linspace(0+1/bz1/2, 1-1/bz1/2, bz1).to(devices[0]) - center[0]
    y_axis = torch.linspace(0+1/bz2/2, 1-1/bz2/2, bz2).to(devices[0]) - center[1]
    xx, yy = torch.meshgrid(x_axis, y_axis)
    
    data1 = torch.cat((torch.unsqueeze(xx.reshape(-1),-1),torch.unsqueeze(yy.reshape(-1),-1)),1)
    data1 = torch.unsqueeze(data1,1)
    data2 = torch.transpose(data1,1,2)
    #print(data1.shape,data2.shape,B.shape)
    kernel = torch.matmul(data1,B)
    kernel = torch.matmul(kernel,data2)
    #print(t.shape)
    kernel = torch.exp(-0.5 * kernel)
    kernel = torch.squeeze(kernel)
    
    return kernel.reshape(-1)

# %%
def gaussian_2B(center, B,bz1 ,bz2):
    B = B.float()
    x_axis = torch.linspace(0+1/bz1/2, 1-1/bz1/2, bz1).to(devices[0]) - center[0]
    y_axis = torch.linspace(0+1/bz2/2, 1-1/bz2/2, bz2).to(devices[0]) - center[1]
    xx, yy = torch.meshgrid(x_axis, y_axis)
    
    data1 = torch.cat((torch.unsqueeze(xx.reshape(-1),-1),torch.unsqueeze(yy.reshape(-1),-1)),1)
    data1 = torch.unsqueeze(data1,1)
    data2 = torch.transpose(data1,1,2)
    #print(data1.shape,data2.shape,B.shape)
    kernel = torch.matmul(data1,B.transpose(0,1))
    kernel = torch.matmul(kernel,B)
    kernel = torch.matmul(kernel,data2)
    #print(t.shape)
    kernel = torch.exp(-0.5 * kernel)
    kernel = torch.squeeze(kernel)
    
    return kernel.reshape(-1)

# %%
def Decoder_B(u,a,B,p,bz1,bz2):
    n_seg = u.shape[0]
    K_init = u.shape[1]
    k = torch.randn(n_seg,K_init,bz1*bz2, dtype=torch.float).to(devices[0])
    g = torch.randn(n_seg,K_init,bz1*bz2, dtype=torch.float).to(devices[0])
    #print('k shape = {}, g shape = {}, batch_size = {}, block index = {}'.format(k.shape,g.shape,batch_size,block_index))
    for n_s in range(n_seg):
        for i in range(K_init):
            k[n_s][i] = gaussian_B(center = u[n_s][i], B = torch.tril(B[n_s][i]),bz1=bz1,bz2=bz2)#torch.tril(self.B[n_b][i]))#
           
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
    z = z.reshape(n_seg,bz1,bz2)

    return z

# %%
def Decoder_2B(u,a,B,p,bz1,bz2):
    n_seg = u.shape[0]
    K_init = u.shape[1]
    k = torch.randn(n_seg,K_init,bz1*bz2, dtype=torch.float).to(devices[0])
    g = torch.randn(n_seg,K_init,bz1*bz2, dtype=torch.float).to(devices[0])
    #print('k shape = {}, g shape = {}, batch_size = {}, block index = {}'.format(k.shape,g.shape,batch_size,block_index))
    for n_s in range(n_seg):
        for i in range(K_init):
            k[n_s][i] = gaussian_2B(center = u[n_s][i], B = torch.tril(B[n_s][i]),bz1=bz1,bz2=bz2)#torch.tril(self.B[n_b][i]))#
           
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
    z = z.reshape(n_seg,bz1,bz2)

    return z

# %%
#zz[maps[0].mean(2) < 20] = 0
imo_file = os.path.join('data','img',partition)
imnames = sorted(os.listdir(imo_file))

time_per_image = torch.zeros(len(imnames))

#print(imnames,uss)
for j,imname in enumerate(imnames):
    
    for i,t in enumerate(times):
    
        name = names[i]
        #print(name,imname,name[:len(imname)])
        if name[:len(imname[:-4])] == imname[:-4]:
            time_per_image[j] = np.max([time_per_image[j],t.sum()])
print(time_per_image)

import pickle
f,ax = plt.subplots(2,8,figsize=(25,5))

print(osizes)
for img_id,imname in enumerate(sorted(os.listdir(imo_file))):
    imo = Image.open(os.path.join(imo_file,imname))
    imo = np.array(imo.resize((imo.size[0],imo.size[1])).convert('L'))
    print(imname, imo.shape)

    bz1 = imo.shape[0]
    bz2 = imo.shape[1]
    print(bz1,bz2)

    nus = torch.tensor([]).to(devices[0])
    nas = torch.tensor([]).to(devices[0])
    nBs = torch.tensor([]).to(devices[0])
    nps = torch.tensor([]).to(devices[0])
    
    for i,n in enumerate(names):
        if n[:len(imname[:-4])] != imname[:-4]:
            continue
        
        u = uss[i].clone()
        B = Bss[i].clone()
        a = ass[i].clone()
        if regular != 'no':
            p = pss[i].clone()
        # create map
        xs,ys = np.where(maps[i] > 20)
        
        indexs = xs * bz_seg + ys
        xt = (u[0,:,0].cpu().detach().numpy()*bz_seg).round()
        yt = (u[0,:,1].cpu().detach().numpy()*bz_seg).round()
        indext = xt * bz_seg + yt
        if K_init > 5:
            matchi = []
            for ind in indext:
                if ind in indexs:
                    matchi.append(True)
                else:
                    matchi.append(False)
            matchi = torch.tensor(matchi).unsqueeze(0)
        else:
            matchi = (torch.ones(a.shape[0]) > 0)
        #print(matchi)
        u = u[matchi].reshape(1,-1,2)
        B = B[matchi].reshape(1,-1,2,2)
        a = a[matchi].reshape(1,-1)
        if regular != 'no':
            p = p[matchi].reshape(1,-1)
            if regular == 'l0':
                mask = torch.sigmoid(p)*(args.lam-args.gamma)+args.gamma
                mask[mask<=0] = torch.tensor(0)
                mask[mask>=1] = torch.tensor(1)
            else:
                mask = torch.zeros(p.shape).to(devices[0])
                mask[p > 0] = p[p > 0]
            u = u[mask>0].reshape(1,-1,2)
            B = B[mask>0].reshape(1,-1,2,2)
            a = a[mask>0].reshape(1,-1)
            p = p[mask>0].reshape(1,-1)
        
        # L,V = torch.linalg.eigh(torch.matmul(B[0].transpose(1,2),B[0]))
        # E = torch.zeros(u.shape[1],2,2).to(devices[0]).float()
        # E[:,0,0] = L[:,0] #/ (osizes[0] / 64)
        # E[:,1,1] = L[:,1] #/ (osizes[0] / 64)
        # newB = torch.matmul(V.transpose(1,2).float(),E.float())
        # newB = torch.matmul(newB.float(),V.float())
        # newB = torch.matmul(B[0].transpose(1,2),B[0]).transpose(1,2)
        B = B  / np.sqrt(osizes[i] / bz_seg)
        print(B.shape)
        B[:,:,0,0] = B[:,:,0,0] / np.sqrt(osizes[i]/bz1)
        B[:,:,1,1] = B[:,:,1,1] / np.sqrt(osizes[i]/bz2)
        B[:,:,1,0] = B[:,:,1,0] / np.sqrt(osizes[i]/np.sqrt(bz1)/np.sqrt(bz2))
        B[:,:,0,1] = B[:,:,0,1] / np.sqrt(osizes[i]/np.sqrt(bz1)/np.sqrt(bz2))
        # B[0][:,0,0] = B[0][:,0,0]  / np.sqrt(osizes[0] / 64)
        # B[0][:,1,0] = B[0][:,0,0]  / np.sqrt(osizes[0] / 64)
        # B[0][:,1,1] = B[0][:,1,1]  / np.sqrt(osizes[0] / 64)
        u[0,:,0] = ((u[0,:,0] * bz_seg) *  (osizes[i] / bz_seg) + x1s[i]) / bz1
        u[0,:,1] = ((u[0,:,1] * bz_seg) *  (osizes[i] / bz_seg) + y1s[i]) / bz2

        nus = torch.cat((nus,u.float()),1)
        nBs = torch.cat((nBs,B.float()),1)
        nas = torch.cat((nas,a.float()),1)
        if regular != 'no':
            nps = torch.cat((nps,p.float()),1)
        #print(nus.shape[1],nps.shape[1])
        #print(u.shape)
        #np.save('{}.npy'.format(imname),{'u':nus,'a':nas,'B':nBs})
    #print(nus.shape)
    with open(os.path.join(save_path,'{}.pkl'.format(imname[:-4])), 'wb') as fp:
        pickle.dump({'u':nus,'a':nas,'B':nBs, 'p':nps}, fp)
        #np.save('{}.npy'.format(imname),{'u':nus,'a':nas,'B':nBs})


    z_B = Decoder_2B(nus,nas,nBs.reshape(1,-1,2,2),nps,bz1,bz2)
    
    if img_id < 8:
        ax[0,img_id].imshow(z_B[0].cpu().detach().numpy()[:bz1,:bz2],cmap='gray',vmin=0,vmax=1)
        psnr = peak_signal_noise_ratio(z_B[0].cpu().detach().numpy()[:bz1,:bz2],imo/255.,data_range=1.0)
        ssim = structural_similarity(z_B[0].cpu().detach().numpy()[:bz1,:bz2],imo/255.,data_range=1.0)
        ax[0,img_id].set_title('psnr: {:.2f} ssim: {:.2f}\n{:.2f} n. of k.{}'.format(psnr,ssim,time_per_image[img_id],nus.shape[1]))
        ax[1,img_id].imshow(imo,cmap='gray')
        ax[0,img_id].axis('off')
        ax[1,img_id].axis('off')
    f.savefig(os.path.join(save_path,'init_results.jpg'))
    #plt.imshow(z_B[0].cpu().detach().numpy(),cmap='gray')
#plt.scatter(yy[matchi],xx[matchi],cmap='gray')

