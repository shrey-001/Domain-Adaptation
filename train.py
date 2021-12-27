import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import numpy as np
import datetime
import os, sys
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, random_split
import pandas as pd
import numpy as np
import glob
from PIL import Image
from torchvision.io import read_image

import models.dann as dann
from data.HGMDataset import HGM 
from data.transforms.HGM_transforms import transform

def get_lambda(epoch, max_epoch):
    p = epoch / max_epoch
    return 2. / (1+np.exp(-10.*p)) - 1.
def sample_view(step, n_batches):
    global view2_set
    if step % n_batches == 0:
        view2_set = iter(loader[1])
    return view2_set.next()


max_epoch=10
MODEL_NAME = 'DANN'
img_dir='data'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


cams=['Left','Right','below',"Front"]
transform=transform()
dataset_loaders=[HGM(i+'_CAM.csv',img_dir,transform) for i in cams]
loader= [DataLoader(train_dataset,sampler=RandomSampler(train_dataset),batch_size=16,num_workers=16,drop_last=True) for train_dataset in dataset_loaders]


F = dann.FeatureExtractor().to(DEVICE)
C =dann.Classifier().to(DEVICE)
D = dann.Discriminator().to(DEVICE)

F_opt = torch.optim.Adam(F.parameters())
C_opt = torch.optim.Adam(C.parameters())
D_opt = torch.optim.Adam(D.parameters())

bce = nn.BCELoss()
xe = nn.CrossEntropyLoss()


step = 0
n_critic = 1 # for training more k steps about Discriminator
n_batches = len(loader[0])//16
# lamda = 0.01

batch_size=16

D_src = torch.ones(batch_size, 1).to(DEVICE) # Discriminator Label to real (16)-->1
D_tgt = torch.zeros(batch_size, 1).to(DEVICE) # Discriminator Label to fake(16)-->0
D_labels = torch.cat([D_src, D_tgt], dim=0) #(32,1)


view2_set = iter(loader[1]) #(16,28,28)



"""# Custom Image Dataset """

ll_c=[]
ll_d=[]
acc_lst=[]

for epoch in range(1, max_epoch+1):
    for idx, (src_images, labels) in enumerate(loader[0]): #(16,1,28,28) and (16)
        # print(src_images.size(),labels.size())
        # exit(0)
        tgt_images, _ = sample_view(step, n_batches)  #16,1,28,28
        # print(tgt_images.size())
        # Training Discriminator
        src, labels, tgt = src_images.to(DEVICE), labels.to(DEVICE), tgt_images.to(DEVICE)
        
        x = torch.cat([src, tgt], dim=0) 
        # print(x.size())
        # exit(0)
        h = F(x)
        # print(h.size())
        # exit(0)
        y = D(h.detach())
        # print(y.size())
        # exit(0)
        Ld = bce(y, D_labels)
        # print(Ld)
        # exit(0)
        D.zero_grad()
        Ld.backward()
        D_opt.step()
        
        
        c = C(h[:batch_size]) #32,512
        y = D(h)
        Lc = xe(c, labels)
        Ld = bce(y, D_labels)
        lamda = 0.1*get_lambda(epoch, max_epoch)
        Ltot = Lc -lamda*Ld
        
        
        F.zero_grad()
        C.zero_grad()
        D.zero_grad()
        
        Ltot.backward()
        
        C_opt.step()
        F_opt.step()
        
        if step % 100 == 0:
            dt = datetime.datetime.now().strftime('%H:%M:%S')
            print('Epoch: {}/{}, Step: {}, D Loss: {:.4f}, C Loss: {:.4f}, lambda: {:.4f} ---- {}'.format(epoch, max_epoch, step, Ld.item(), Lc.item(), lamda, dt))
            ll_c.append(Lc)
            ll_d.append(Ld)
        
        if step % 300 == 0:
            F.eval()
            C.eval()
            with torch.no_grad():
                corrects = torch.zeros(1).to(DEVICE)
                for idx, (src, labels) in enumerate(loader[0]):
                    src, labels = src.to(DEVICE), labels.to(DEVICE)
                    c = C(F(src)) #(16,26)
                    _, preds = torch.max(c, 1) #16
                    corrects += (preds == labels).sum()
                acc = corrects.item() / len(loader[0].dataset)
                print('***** Eval Result: {:.4f}, Step: {}'.format(acc, step))
                
                corrects = torch.zeros(1).to(DEVICE)
                for idx, (tgt, labels) in enumerate(loader[1]):
                    tgt, labels = tgt.to(DEVICE), labels.to(DEVICE)
                    c = C(F(tgt))
                    _, preds = torch.max(c, 1)
                    corrects += (preds == labels).sum()
                acc = corrects.item() / len(loader[1].dataset)
                print('***** Test Result: {:.4f}, Step: {}'.format(acc, step))
                acc_lst.append(acc)
                
            F.train()
            C.train()
        step += 1

