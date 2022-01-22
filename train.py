import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import numpy as np
import datetime
import os, sys
import shutil
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, random_split
import pandas as pd
import numpy as np
import glob
from PIL import Image
from torchvision.io import read_image
import torch.optim as optim
from utils import AverageMeter, accuracy
from models.mobilenetv2 import mobilenet_v2

import models.dann as dann
from data.HGMDataset import HGM 
from data.transforms.HGM_transforms import transform_resnet,transform_dummy, transform_test
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
batch_size=16


transform=transform_resnet()
transform_te=transform_test()



train_loaders=[HGM(i+'_CAM.csv',img_dir+'train',transform) for i in cams]
train_loader= [DataLoader(train_dataset,sampler=RandomSampler(train_dataset),batch_size=batch_size,num_workers=8,drop_last=True) for train_dataset in train_loaders]
test_loaders=[HGM(i+'_CAM.csv',img_dir+'val',transform_te) for i in cams]
test_loader= [DataLoader(train_dataset,sampler=RandomSampler(train_dataset),batch_size=batch_size,num_workers=8,drop_last=True) for train_dataset in test_loaders]

wandb.login(key="1f50b56189ad0287617289acd72127489c7fe801")
config={"model_name":MODEL_NAME,"Batch_size":batch_size,"lr":1e-3}
#wandb.init(project="domain_adaptation",entity="shreyanshsaxena",name="base_experiment-split-80-seed-137-12-classes",config=config)
#simple - dummy
#simple- resnet
#dann -resnet
def save_checkpoint(state, is_best, checkpoint):
    filename = f'checkpoint.pth.tar'
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, f'model_best.pth.tar'))


def num_parameters(model):
    for s,p in model.named_parameters():
             print(s,':',p.numel())
    print("Total Parameters:",sum(p.numel() for p in model.parameters()))
    print("Trainable Parameters:",sum(p.numel() for p in model.parameters() if p.requires_grad))

def mobilenet_simple_cla():
    base='/raid/dhruv_g_ch/saved_models'
    model_out='mobilenet_True_Augumented'
    out_dir=os.path.join(base,model_out)
    os.makedirs(out_dir, exist_ok=True)

    wandb.init(project="domain_adaptation",entity="shreyanshsaxena",name="simple-mobile2",config=config)
    model=mobilenet_v2(pretrained=True,progress=True,num_classes=13).to(DEVICE)
    num_parameters(model)
    model_opt = torch.optim.Adam(model.parameters())
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(model_opt,mode='min', factor=0.1, patience=10)
    xe = nn.CrossEntropyLoss()
    step = 0
    best_acc=0

    n_batches = len(train_loader[0])//batch_size

    view2_set = iter(train_loader[1])
    
    for epoch in range(1, max_epoch+1):
        losses_source_train=AverageMeter()
        losses_source_test=AverageMeter()

        corrects_t = torch.zeros(1).to(DEVICE)
        for idx, (src_images, labels) in enumerate(train_loader[0]): #(16,1,28,28) and (16)
            
            src, labels = src_images.to(DEVICE), labels.to(DEVICE)
            
        
            c = model(src)
            
            _, preds = torch.max(c, 1) 
            corrects_t += (preds == labels).sum()
        
            Lc = xe(c, labels)
            losses_source_train.update(Lc.item())
        
                
            model.zero_grad()
           
            
            Lc.backward()
            
            model_opt.step()
    
            step=step+1
            
        
        dt = datetime.datetime.now().strftime('%H:%M:%S')
        print('Epoch: {}/{}, Step: {}, C Loss: {:.4f}, C Accuracy: {:.4f}, time: {}'.format(epoch, max_epoch, step, losses_source_train.avg,corrects_t.item() / len(train_loader[0].dataset), dt))
        
                
        model.eval()
        
        with torch.no_grad():
            corrects = torch.zeros(1).to(DEVICE)
            for idx, (src, labels) in enumerate(test_loader[0]):

                src, labels = src.to(DEVICE), labels.to(DEVICE)
                
                c = model(src)
                Lc = xe(c, labels)
                losses_source_test.update(Lc.item())
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
        scheduler.step(losses_source_test.avg)
            
        is_best = acc_source_test > best_acc
        best_acc = max(acc_source_test, best_acc)
        model_to_save = model
        state_dict_to_save = model_to_save.state_dict()
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': state_dict_to_save,
                'acc': acc_source_test,
                'best_acc': best_acc,
                'optimizer': model_opt.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best, out_dir)
    
        wandb.log({"loss_classifier":losses_source_train.avg,"Accuracy_source_test":acc_source_test,"Accuracy_target_test":acc_target_test,"Accuracy_source_train":corrects_t.item()/len(train_loader[0].dataset),"Accuracy_target_train":acc_target_train})

                    
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