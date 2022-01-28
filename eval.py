import torch
import torch.nn as nn
from data import get_dataloaders 
from utils import AverageMeter,get_accuracy
from data.transforms import transform_train,transform_dummy,transform_test 
from pathlib import Path
import os
from models.mobilenetv2 import mobilenet_v2
from utils import load_checkpoint
import matplotlib.pyplot as plt
import numpy as np


DEVICE = "cpu"
cams=['Left_CAM.csv','Right_CAM.csv','Below_CAM.csv',"Front_CAM.csv"]
base_dir: Path = Path(__file__).parents[0]
data_dir = base_dir / '../csv_list'
train_dir = data_dir / 'seed-137/train'
val_dir = data_dir / 'seed-137/val'


model_out='mobilenet_True_Augumented'
base='/raid/dhruv_g_ch/saved_models'
num_classes = 13
def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    print('here3')
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    #plt.figure(figsize=(60,60))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    #plt.xticks(tick_marks, classes, rotation=45)
    #plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('1.png')
def confusion_matrix(dataloader,model,num_classes):

    confusion_matrix = torch.zeros(num_classes, num_classes)
    with torch.no_grad():
        for inputs, classes in dataloader:
            inputs = inputs.to(DEVICE)
            classes = classes.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for t, p in zip(classes.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

    print(confusion_matrix)
    print(confusion_matrix.diag()/confusion_matrix.sum(1))
    cnf=confusion_matrix
    classes=[i for i in range(num_classes)]
    plot_confusion_matrix(cnf.numpy(), classes=classes,title='Confusion matrix, with normalization')

def evaluate(model,dataloader,loss_function,need_loss=False):
    """
        Returns:
            Accuracies and loss
    """
    # Enable evaluation mode
    model.eval()
    with torch.no_grad():
        # Running average class
        accuracy = AverageMeter()
        if need_loss:
            loss = AverageMeter()

        # Evaluation
        for data_batch, labels_batch in dataloader:

            # load data to Device
            data_batch, labels_batch = data_batch.to(DEVICE), labels_batch.to(DEVICE)
            # model output 
            output = model(data_batch)
            _, output_batch = torch.max(output, 1)
            # loss and accuracy
            if need_loss:
                batch_loss = loss_function(output, labels_batch)
            batch_accuracy = get_accuracy(output_batch,labels_batch)
            # update loss and accuracy meter
            accuracy.update(batch_accuracy)
            if need_loss:
                loss.update(batch_loss.item())
    if need_loss:
        return accuracy.avg, loss.avg
    else:
        return accuracy.avg

if __name__ == '__main__':

    model_dir = os.path.join(base,model_out)
    camera_view = 0
    restore_file = 'model_best' #' last'
    #os.path.join(args.model_dir, args.restore_file + '.pth.tar')
    checkpoint = os.path.join(model_dir, restore_file + '.pth.tar')

    dl1, dl2 = get_dataloaders(cams[camera_view],train_dir,val_dir,train_transform=transform_train,val_transform=transform_test)

    model=mobilenet_v2(pretrained=True,progress=True,num_classes=num_classes).to(DEVICE)
    loss_function = nn.CrossEntropyLoss()
    print(torch.cuda.is_available())
    #print(model)

    load_checkpoint(checkpoint, model)
    #accuracy = evaluate(model,dl2,loss_function)
    #print(accuracy)
    confusion_matrix(dl2,model,num_classes)


    


