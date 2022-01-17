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
from models.mobilenetv2 import mobilenet_v2

import models.dann as dann
from data.HGMDataset import HGM 
from data.transforms.HGM_transforms import transform_resnet,transform_dummy
from data.transforms.simple_transforms import transform
import wandb
import gc
gc.collect()
torch.cuda.empty_cache()

def get_lambda(epoch, max_epoch):
    p = epoch / max_epoch
    return 2. / (1+np.exp(-10.*p)) - 1.
def sample_view(step, n_batches):
    global view2_set
    if step % n_batches == 0:
        view2_set = iter(train_loader[1])
    return view2_set.next()

n_critic = 1
max_epoch=300
MODEL_NAME = 'DANN'
img_dir='../csv_list/seed-137/'
DEVICE = torch.device("cuda")
# print(torch.cuda.is_available())
cams=['Left','Right','Below',"Front"]
batch_size=4


transform=transform_resnet()



train_loaders=[HGM(i+'_CAM.csv',img_dir+'train',transform) for i in cams]
train_loader= [DataLoader(train_dataset,sampler=RandomSampler(train_dataset),batch_size=batch_size,num_workers=8,drop_last=True) for train_dataset in train_loaders]
test_loaders=[HGM(i+'_CAM.csv',img_dir+'val',transform) for i in cams]
test_loader= [DataLoader(train_dataset,sampler=RandomSampler(train_dataset),batch_size=batch_size,num_workers=8,drop_last=True) for train_dataset in test_loaders]

wandb.login(key="1f50b56189ad0287617289acd72127489c7fe801")
config={"model_name":MODEL_NAME,"Batch_size":batch_size,"lr":1e-3}
#wandb.init(project="domain_adaptation",entity="shreyanshsaxena",name="base_experiment-split-80-seed-137-12-classes",config=config)
#simple - dummy
#simple- resnet
#dann -resnet
def mobilenet_simple_cla():
    wandb.init(project="domain_adaptation",entity="shreyanshsaxena",name="simple-mobile",config=config)
    model=mobilenet_v2(pretrained=False,progress=True,num_classes=13).to(DEVICE)
    model_opt = torch.optim.Adam(model.parameters())
    xe = nn.CrossEntropyLoss()
    step = 0

    n_batches = len(train_loader[0])//batch_size

    view2_set = iter(train_loader[1])
    for epoch in range(1, max_epoch+1):

        corrects_t = torch.zeros(1).to(DEVICE)
        for idx, (src_images, labels) in enumerate(train_loader[0]): #(16,1,28,28) and (16)
            
            src, labels = src_images.to(DEVICE), labels.to(DEVICE)
            
        
            c = model(src)
            
            _, preds = torch.max(c, 1) 
            corrects_t += (preds == labels).sum()
        
            Lc = xe(c, labels)
        
                
            model.zero_grad()
           
            
            Lc.backward()
            
            model_opt.step()
    
            step=step+1
            
        
        dt = datetime.datetime.now().strftime('%H:%M:%S')
        print('Epoch: {}/{}, Step: {}, C Loss: {:.4f}, C Accuracy: {:.4f}, time: {}'.format(epoch, max_epoch, step, Lc.item(),corrects_t.item() / len(train_loader[0].dataset), dt))
        
                
        model.eval()
        
        with torch.no_grad():
            corrects = torch.zeros(1).to(DEVICE)
            for idx, (src, labels) in enumerate(test_loader[0]):
                src, labels = src.to(DEVICE), labels.to(DEVICE)
                c = model(src)
                # print(c.size()) #(16,26)
                _, preds = torch.max(c, 1) #16
                corrects += (preds == labels).sum()
            acc_source_test = corrects.item() / len(test_loader[0].dataset)
            print('* Source_test: {:.4f}, Step: {}'.format(acc_source_test, step))
            corrects = torch.zeros(1).to(DEVICE)
            for idx, (tgt, labels) in enumerate(test_loader[1]):
                tgt, labels = tgt.to(DEVICE), labels.to(DEVICE)
                c = model(tgt)
                _, preds = torch.max(c, 1)
                corrects += (preds == labels).sum()
            acc_target_test = corrects.item() / len(test_loader[1].dataset)
            print('* Target_Test: {:.4f}, Step: {}'.format(acc_target_test, step))
            #acc_lst.append(acc_target_test)

            for idx, (tgt, labels) in enumerate(train_loader[1]):
                tgt, labels = tgt.to(DEVICE), labels.to(DEVICE)
                c = model(tgt)
                _, preds = torch.max(c, 1)
                corrects += (preds == labels).sum()
            acc_target_train = corrects.item() / len(train_loader[1].dataset)
            print('* Target_Train Result: {:.4f}, Step: {}'.format(acc_target_train, step))
            #acc_lst.append(acc_target_test)

        
        wandb.log({"loss_classifier":Lc,"Accuracy_source_test":acc_source_test,"Accuracy_target_test":acc_target_test,"Accuracy_source_train":corrects_t.item()/len(train_loader[0].dataset),"Accuracy_target_train":acc_target_train})

                    
        model.train()
        
def simple_classification():
    wandb.init(project="domain_adaptation",entity="shreyanshsaxena",name="simple-dummy",config=config)

    F = dann.FeatureExtractor().to(DEVICE)
    C =dann.Classifier().to(DEVICE)

    F_opt = torch.optim.Adam(F.parameters())
    C_opt = torch.optim.Adam(C.parameters())

    xe = nn.CrossEntropyLoss()

    step = 0

    n_batches = len(train_loader[0])//batch_size

    view2_set = iter(train_loader[1])

    for epoch in range(1, max_epoch+1):

        corrects_t = torch.zeros(1).to(DEVICE)
        for idx, (src_images, labels) in enumerate(train_loader[0]): #(16,1,28,28) and (16)
            
            src, labels = src_images.to(DEVICE), labels.to(DEVICE)
            
        
            h = F(src)
            
            c = C(h) #16,512
            _, preds = torch.max(c, 1) 
            corrects_t += (preds == labels).sum()
        
            Lc = xe(c, labels)
        
                
            F.zero_grad()
            C.zero_grad()
            #D.zero_grad()
            
            Lc.backward()
            
            C_opt.step()
            F_opt.step()
            step=step+1
            
        
        dt = datetime.datetime.now().strftime('%H:%M:%S')
        print('Epoch: {}/{}, Step: {}, C Loss: {:.4f}, C Accuracy: {:.4f},'.format(epoch, max_epoch, step, Lc.item(),corrects_t.item() / len(train_loader[0].dataset), dt))
        
                
        F.eval()
        C.eval()
        with torch.no_grad():
            corrects = torch.zeros(1).to(DEVICE)
            for idx, (src, labels) in enumerate(test_loader[0]):
                src, labels = src.to(DEVICE), labels.to(DEVICE)
                c = C(F(src))
                # print(c.size()) #(16,26)
                _, preds = torch.max(c, 1) #16
                corrects += (preds == labels).sum()
            acc_source_test = corrects.item() / len(test_loader[0].dataset)
            print('* Source_test: {:.4f}, Step: {}'.format(acc_source_test, step))
            corrects = torch.zeros(1).to(DEVICE)
            for idx, (tgt, labels) in enumerate(test_loader[1]):
                tgt, labels = tgt.to(DEVICE), labels.to(DEVICE)
                c = C(F(tgt))
                _, preds = torch.max(c, 1)
                corrects += (preds == labels).sum()
            acc_target_test = corrects.item() / len(test_loader[1].dataset)
            print('* Target_Test: {:.4f}, Step: {}'.format(acc_target_test, step))
            #acc_lst.append(acc_target_test)

            for idx, (tgt, labels) in enumerate(train_loader[1]):
                tgt, labels = tgt.to(DEVICE), labels.to(DEVICE)
                c = C(F(tgt))
                _, preds = torch.max(c, 1)
                corrects += (preds == labels).sum()
            acc_target_train = corrects.item() / len(train_loader[1].dataset)
            print('* Target_Train Result: {:.4f}, Step: {}'.format(acc_target_train, step))
            #acc_lst.append(acc_target_test)

        
        wandb.log({"loss_classifier":Lc,"Accuracy_source_test":acc_source_test,"Accuracy_target_test":acc_target_test,"Accuracy_source_train":corrects_t.item()/len(train_loader[0].dataset),"Accuracy_target_train":acc_target_train})

                    
        F.train()
        C.train()

def DANN():
    wandb.init(project="domain_adaptation",entity="shreyanshsaxena",name="DANN-resnet",config=config)

    F = dann.FeatureExtractor().to(DEVICE)
    C =dann.Classifier().to(DEVICE)
    D = dann.Discriminator().to(DEVICE)


    F_opt = torch.optim.Adam(F.parameters())
    C_opt = torch.optim.Adam(C.parameters())
    D_opt = torch.optim.Adam(D.parameters())

    bce = nn.BCELoss()
    xe = nn.CrossEntropyLoss()


    step = 0
    

    n_batches = len(train_loader[0])//batch_size
    # lamda = 0.01


    


    D_src = torch.ones(batch_size, 1).to(DEVICE) # Discriminator Label to real (16)-->1
    D_tgt = torch.zeros(batch_size, 1).to(DEVICE) # Discriminator Label to fake(16)-->0
    D_labels = torch.cat([D_src, D_tgt], dim=0) #(32,1)


    view2_set = iter(train_loader[1]) #(16,28,28)



    # Custom Image Dataset

    ll_c=[]
    ll_d=[]
    acc_lst=[]
    

    for epoch in range(1, max_epoch+1):
        accuracy_dis=0
        corrects_t = torch.zeros(1).to(DEVICE)
        for idx, (src_images, labels) in enumerate(train_loader[0]): #(16,1,28,28) and (16)
            #print(len(train_loader[0].dataset))
            #print(src_images.size(),labels.size())
            #exit(0)
            tgt_images, _ = sample_view(step, n_batches)  #16,1,28,28
            #print(tgt_images.size())
            # Training Discriminator
            src, labels, tgt = src_images.to(DEVICE), labels.to(DEVICE), tgt_images.to(DEVICE)
            
            x = torch.cat([src, tgt], dim=0) 
            #print(x.size())
            # exit(0)
            h = F(x)
            #print(h.size())
            # exit(0)
            y = D(h.detach())
            #print(y.size())
            # exit(0)
            Ld = bce(y, D_labels)
            #print(Ld)
            #exit(0)
            D.zero_grad()
            Ld.backward()
            D_opt.step()
            
            
            c = C(h[:batch_size]) #16,512
            _, preds = torch.max(c, 1) 
            corrects_t += (preds == labels).sum()
            y = D(h)
            accuracy_dis+=y.mean().item()
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
            step=step+1
            
        
        dt = datetime.datetime.now().strftime('%H:%M:%S')
        print('Epoch: {}/{}, Step: {}, D Loss: {:.4f}, C Loss: {:.4f}, C Accuracy: {:.4f}, lambda: {:.4f} ---- {}'.format(epoch, max_epoch, step, Ld.item(), Lc.item(),corrects_t.item() / len(train_loader[0].dataset),lamda, dt))
        ll_c.append(Lc)
        ll_d.append(Ld)
                
            
            
        F.eval()
        C.eval()
        D.eval()
        with torch.no_grad():
            corrects = torch.zeros(1).to(DEVICE)
            for idx, (src, labels) in enumerate(test_loader[0]):
                src, labels = src.to(DEVICE), labels.to(DEVICE)
                c = C(F(src))
                # print(c.size()) #(16,26)
                _, preds = torch.max(c, 1) #16
                corrects += (preds == labels).sum()
            acc_source_test = corrects.item() / len(test_loader[0].dataset)
            print('* Eval Result: {:.4f}, Step: {}'.format(acc_source_test, step))
            corrects = torch.zeros(1).to(DEVICE)
            for idx, (tgt, labels) in enumerate(test_loader[1]):
                tgt, labels = tgt.to(DEVICE), labels.to(DEVICE)
                c = C(F(tgt))
                _, preds = torch.max(c, 1)
                corrects += (preds == labels).sum()
            acc_target_test = corrects.item() / len(test_loader[1].dataset)
            print('* Test Result: {:.4f}, Step: {}'.format(acc_target_test, step))
            acc_lst.append(acc_target_test)

        wandb.log({"loss_discriminator":Ld,"loss_classifier":Lc,"Accuracy_source_test":acc_source_test,"Accuracy_target_test":acc_target_test,"Accuracy_disriminator":accuracy_dis/len(train_loader[0]),"Accuracy_source_train":corrects_t.item()/len(train_loader[0].dataset)})

                    
        F.train()
        D.train()
        C.train()

# simple_classification()
mobilenet_simple_cla()