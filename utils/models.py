import numpy

def num_parameters(model):
    for s,p in model.named_parameters():
             print(s,':',p.numel())
    print("Total Parameters:",sum(p.numel() for p in model.parameters()))
    print("Trainable Parameters:",sum(p.numel() for p in model.parameters() if p.requires_grad))
def get_lambda(epoch, max_epoch):
    p = epoch / max_epoch
    return 2. / (1+np.exp(-10.*p)) - 1.