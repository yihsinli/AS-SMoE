import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
import os
import argparse

#from torchvision import transforms as T
from PIL import Image, ImageOps
from datetime import datetime
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
#from piqa import SSIM
#import pytorch_ssim
#from AE import AE
from scipy.fftpack import dct, idct
from sklearn.cluster import KMeans
import csv
#import pandas as pd
import pickle
import glob
import scipy.io as sio
import numpy as np
from skimage import color

parser = argparse.ArgumentParser()
#parser.add_argument('--name', type=str, default="img_lena_crop_2")
parser.add_argument('--partition','--p', type=str, default="val")
parser.add_argument('--diff', type=str, default="20")
parser.add_argument('--init', type=str, default="mdbscan")
parser.add_argument('--bz_seg', type=int, default=32)
parser.add_argument('--margin', type=int, default=10)
args = parser.parse_args()

partition = args.partition
diff = args.diff
init = args.init
bz_seg = args.bz_seg

map_file = './data/seg/{}/{}/{}'.format(init,diff,partition)
img_file = './data/img/{}'.format(partition)
#files = sorted(glob.glob(os.path.join(map_file,'*.mat')))
image_names = sorted(os.listdir(img_file))

print(os.path.join(init))
# if not os.path.isdir(os.path.join(init)):
#     os.mkdir(os.path.join(init))
# if not os.path.isdir(os.path.join(init,diff)):
#     os.mkdir(os.path.join(init,diff))
# if not os.path.isdir(os.path.join(init,diff,'{}x{}-img'.format(bz_seg,bz_seg))):
#     os.mkdir(os.path.join(init,diff,'{}x{}-img'.format(bz_seg,bz_seg)))
# if not os.path.isdir(os.path.join(init,diff,'{}x{}-seg'.format(bz_seg,bz_seg))):
#     os.mkdir(os.path.join(init,diff,'{}x{}-seg'.format(bz_seg,bz_seg)))
# if not os.path.isdir(os.path.join(init,diff,'{}x{}-mat'.format(bz_seg,bz_seg))):
#     os.mkdir(os.path.join(init,diff,'{}x{}-mat'.format(bz_seg,bz_seg)))
# if not os.path.isdir(os.path.join(init,diff,'{}x{}-img'.format(bz_seg,bz_seg),partition)):
#     os.mkdir(os.path.join(init,diff,'{}x{}-img'.format(bz_seg,bz_seg),partition))
# if not os.path.isdir(os.path.join(init,diff,'{}x{}-seg'.format(bz_seg,bz_seg),partition)):
#     os.mkdir(os.path.join(init,diff,'{}x{}-seg'.format(bz_seg,bz_seg),partition))
# if not os.path.isdir(os.path.join(init,diff,'{}x{}-mat'.format(bz_seg,bz_seg),partition)):
#     os.mkdir(os.path.join(init,diff,'{}x{}-mat'.format(bz_seg,bz_seg),partition))

img_path = os.path.join(init,diff,'{}x{}-img-{}'.format(bz_seg,bz_seg,args.margin),partition)
seg_path = os.path.join(init,diff,'{}x{}-seg-{}'.format(bz_seg,bz_seg,args.margin),partition)
mat_path = os.path.join(init,diff,'{}x{}-mat-{}'.format(bz_seg,bz_seg,args.margin),partition)
os.makedirs(img_path,exist_ok=True)
os.makedirs(seg_path,exist_ok=True)
os.makedirs(mat_path,exist_ok=True)
#for n in np.arange(50,101,10):
images = []
ori_images = []
c = 0
for i,name in enumerate(sorted(image_names)):

    image = []
    ori_image = []

    imname = name.split('/')[-1][:-4]
    label = sio.loadmat(os.path.join(map_file,'{}.mat'.format(imname)))['label']
    
    #img = plt.imread(os.path.join(img_file,'{}.jpg'.format(imname)))
    img = plt.imread(os.path.join(img_file,name))
    if len(img.shape) < 3:
        img = np.expand_dims( img , axis=-1)
    if img.max() < 2:
        img = img*255

    for l in range(1,label.max()+1):
        
        seg = np.zeros(img.shape)
        seg[label==l] = 255
        x = np.where(label==l)[0]
        y = np.where(label==l)[1]
        w = x.max() - x.min()
        h = y.max() - y.min()
        
        tb = np.max([w,h])+args.margin

        x1 = int((x.max()+x.min())/2-tb/2)
        x2 = int((x.max()+x.min())/2+tb/2)
        y1 = int((y.max()+y.min())/2-tb/2)
        y2 = int((y.max()+y.min())/2+tb/2)
        
        crop_im  = img[np.max([0,x1]):np.min([img.shape[0],x2]),np.max([0,y1]):np.min([img.shape[1],y2])]
        crop_seg = seg[np.max([0,x1]):np.min([img.shape[0],x2]),np.max([0,y1]):np.min([img.shape[1],y2])]
        
        tim = np.uint8(np.zeros((tb,tb,img.shape[2])))
        tseg = np.uint8(np.zeros((tb,tb,img.shape[2])))
        tim[np.max([0,x1])-x1:np.min([img.shape[0],x2])-x1,np.max([0,y1])-y1:np.min([img.shape[1],y2])-y1,:] = np.uint8(crop_im)
        tseg[np.max([0,x1])-x1:np.min([img.shape[0],x2])-x1,np.max([0,y1])-y1:np.min([img.shape[1],y2])-y1,:] = np.uint8(crop_seg)

        re_im = np.array(Image.fromarray(tim).resize((bz_seg,bz_seg)))
        re_seg = np.array(Image.fromarray(tseg).resize((bz_seg,bz_seg)))
        
        
        plt.imsave(os.path.join('./{}/{}/{}x{}-img-{}/{}/'.format(init,diff,bz_seg,bz_seg,args.margin,partition),'{}_{:04d}.jpg'.format(imname,l)),np.uint8(re_im))
        plt.imsave(os.path.join('./{}/{}/{}x{}-seg-{}/{}/'.format(init,diff,bz_seg,bz_seg,args.margin,partition),'{}_{:04d}.jpg'.format(imname,l)),np.uint8(re_seg))
        plt.imsave(os.path.join('./{}/{}/{}x{}-mat-{}/{}/'.format(init,diff,bz_seg,bz_seg,args.margin,partition),'{}-{:04d}.jpg'.format(imname,l)),(color.label2rgb(re_seg.mean(2)/255,re_im.mean(2)/255)))
            
        with open('ori_info_{}_{}_{}_{}.csv'.format(init,diff,partition,args.margin),'a',newline='') as f:
            csvwriter = csv.writer(f)
            csvwriter.writerow(['{}_{:04d}'.format(imname,l),x1,y1,x2,y2,tb])
        
        c += 1

    print('{:4d}/{:4d}'.format(i,len(image_names)))
    