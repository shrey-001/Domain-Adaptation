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
from torch.utils.data import DataLoader, SequentialSampler, random_split
import pandas as pd
import numpy as np
import glob
from PIL import Image
from torchvision.io import read_image
import torch.optim as optim
from utils import AverageMeter, num_parameters, get_lambda, save_checkpoint
from models.mobilenetv2 import mobilenet_v2

import models.dann as dann
from data.HGMDataset import HGM
from data.transforms import transform_train,transform_dummy,transform_test  
from trainer.classification import train_epoch
from eval import evaluate

import wandb
import gc

from pathlib import Path


from data import get_dataloaders
gc.collect()
torch.cuda.empty_cache()

############################################################
############        CONSTANTS      #########################
############################################################
wandb_key = "1f50b56189ad0287617289acd72127489c7fe801"
base_dir: Path = Path(__file__).parents[0]
data_dir = base_dir / 'datasets'
train_dir = data_dir / 'seed-137/train'
val_dir = data_dir / 'seed-137/val'
cams=['Left_CAM.csv','Right_CAM.csv','Below_CAM.csv',"Front_CAM.csv"]
############################################################
############        VARIABLE        ########################
############################################################    
batch_size=16
max_epoch=1
lr=1e-3
num_classes=13
MODEL_NAME = 'DANN'
model_out='mobilenet_True_Augumented'
base='/raid/shreyansh_s_ch/saved_models'
############################################################
############################################################


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


wandb.login(key=wandb_key)
config={"model_name":MODEL_NAME,"Batch_size":batch_size,"lr":lr}
wandb.init(project="domain_adaptation",entity="shreyanshsaxena",name=MODEL_NAME,config=config)

def train_and_evaluate():

    out_dir=os.path.join(base,model_out)

    train_loaders, test_loaders = get_dataloaders(cams[0],train_dir,val_dir,train_transform=transform_train,val_transform=transform_test) 
    target_train, target_test = get_dataloaders(cams[1],train_dir,val_dir,train_transform=transform_test,val_transform=transform_test)

    model=mobilenet_v2(pretrained=True,progress=True,num_classes=num_classes).to(DEVICE)
    model_loss_fn = nn.CrossEntropyLoss()
    model_optimizer = torch.optim.Adam(model.parameters())
    model_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer,mode='min',factor = 0.1, patience=10)

    best_acc = 0

    n_batches = len(train_loaders)//batch_size

    for epoch in range(1,max_epoch+1):
        #Training
        source_train_accuracy, source_train_loss = train_epoch(model,train_loaders,model_optimizer,model_loss_fn)
        #Evaluate
        source_test_accuracy, source_test_loss = evaluate(model,test_loaders,model_loss_fn,need_loss=True)
        target_train_accuracy = evaluate(model,target_train,model_loss_fn)
        target_test_accuracy = evaluate(model,target_test,model_loss_fn)
        # Scheduler step
        model_scheduler.step(source_test_loss)
        # Save model
        is_best = source_test_accuracy > best_acc
        best_acc = max(source_test_accuracy, best_acc)
        model_to_save = model
        state_dict_to_save = model_to_save.state_dict()
        save_checkpoint({
                'epoch': epoch,
                'state_dict': state_dict_to_save,
                'acc': source_test_accuracy,
                'best_acc': best_acc,
                'optimizer': model_optimizer.state_dict(),
                'scheduler': model_scheduler.state_dict(),
            }, is_best, out_dir)
        # Wandb loss
        
        wandb.log({"loss_classifier":source_train_loss,
                    "Accuracy_source_test":source_test_accuracy,
                    "Accuracy_target_test":target_test_accuracy,
                    "Accuracy_source_train":source_train_accuracy,
                    "Accuracy_target_train":target_train_accuracy})
    
def mobilenet_simple_cla():
    
    out_dir=os.path.join(base,model_out)

    wandb.init(project="domain_adaptation",entity="shreyanshsaxena",name="simple-mobile2",config=config)

    model=mobilenet_v2(pretrained=True,progress=True,num_classes=num_classes).to(DEVICE)
    num_parameters(model)
    model_opt = torch.optim.Adam(model.parameters())
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(model_opt,mode='min', factor=0.1, patience=10)
    xe = nn.CrossEntropyLoss()
    step = 0
    best_acc=0

    n_batches = len(train_loaders)//batch_size

    view2_set = iter(target_train)
    
    for epoch in range(1, max_epoch+1):
        losses_source_train=AverageMeter()
        losses_source_test=AverageMeter()

        corrects_t = torch.zeros(1).to(DEVICE)
        for idx, (src_images, labels) in enumerate(train_loaders): #(16,1,28,28) and (16)
            
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
        print('Epoch: {}/{}, Step: {}, C Loss: {:.4f}, C Accuracy: {:.4f}, time: {}'.format(epoch, max_epoch, step, losses_source_train.avg,corrects_t.item() / len(train_loaders.dataset), dt))
        
                
        model.eval()
        
        # with torch.no_grad():
        #     accuracy_1 = get_accuracy(model, test_loaders)
        #     print(accuracy_1)
        #     accuracy_4 = get_accuracy(model, test_loaders)
        #     accuracy_2 = get_accuracy(model, target_test)
        #     print(accuracy_2)
        #     accuracy_3 = get_accuracy(model, target_train)
        #     print(accuracy_3)

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
    
        wandb.log({"loss_classifier":losses_source_train.avg,"Accuracy_source_test":acc_source_test,"Accuracy_target_test":acc_target_test,"Accuracy_source_train":corrects_t.item()/len(train_loaders.dataset),"Accuracy_target_train":acc_target_train})

                    
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
train_and_evaluate()