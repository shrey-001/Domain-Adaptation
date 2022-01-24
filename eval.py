import torch
from data import get_dataloaders 
from utils import AverageMeter,get_accuracy
from data.transforms import transform_train,transform_dummy,transform_test 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cams=['Left_CAM.csv','Right_CAM.csv','Below_CAM.csv',"Front_CAM.csv"]
base_dir: Path = Path(__file__).parents[0]
data_dir = base_dir / 'datasets'
train_dir = data_dir / 'seed-137/train'
val_dir = data_dir / 'seed-137/val'
model_out='mobilenet_True_Augumented'
base='/raid/shreyansh_s_ch/saved_models'
num_classes = 13

@torch.no_grad
def evaluate(model,dataloader,loss_function,need_loss=False):
    """
        Returns:
            Accuracies and loss
    """
    # Enable evaluation mode
    model.eval()
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
        batch_accuracy = get_accuracy(out_batch,labels_batch)
        # update loss and accuracy meter
        accuracy.update(batch_accuracy)
        if need_loss:
            loss.update(batch_loss.item())
    if need_loss:
        return accuracy.avg, loss.avg
    else return accuracy.avg

if __name__ == '__main__':

    model_dir = os.path.join(base,model_out)
    camera_view = 1
    restore_file = 'best' #' last'
    #os.path.join(args.model_dir, args.restore_file + '.pth.tar')
    os.path.join(model_dir, restore_file + '.pth.tar')

    dl = get_dataloaders(cams[camera_view],train_dir,val_dir,train_transform=transform_train,val_transform=transform_test)

    model=mobilenet_v2(pretrained=True,progress=True,num_classes=num_classes).to(DEVICE)
    model_loss_fn = nn.CrossEntropyLoss()

    load_checkpoint(checkpoint, model)

    accuracy = evaluate(model,dl,model_loss_fn)


    


