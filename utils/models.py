import numpy
import torch
DEVICE="cuda"
def num_parameters(model):
    for s,p in model.named_parameters():
             print(s,':',p.numel())
    print("Total Parameters:",sum(p.numel() for p in model.parameters()))
    print("Trainable Parameters:",sum(p.numel() for p in model.parameters() if p.requires_grad))
def get_lambda(epoch, max_epoch):
    p = epoch / max_epoch
    return 2. / (1+np.exp(-10.*p)) - 1.
@torch.no_grad()
def get_accuracy(model, dataloaders):
    corrects = torch.zeros(1).to(DEVICE)
    for idx, (src, labels) in enumerate(dataloaders):
        src, labels = src.to(DEVICE), labels.to(DEVICE)
        c = model(src)
        _, preds = torch.max(c, 1)
        corrects += (preds == labels).sum()
    accuracy = corrects.item() / len(dataloaders.dataset)
    print(corrects.item(),len(dataloaders.dataset))
    return accuracy