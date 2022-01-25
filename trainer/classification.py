import torch
from utils import AverageMeter, get_accuracy

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train_epoch(model, dataloader, optimizer, loss_function):
    """
        Returns:
            Accuracies and loss
    """
    # Enable Train Mode
    model.train()
    # Running average class
    accuracy = AverageMeter()
    loss = AverageMeter()

    # Training
    for data_batch, labels_batch in dataloader: #(16,1,28,28) and (16)
        # load data to Device  
        data_batch, labels_batch = data_batch.to(DEVICE), labels_batch.to(DEVICE)
        # model output
        output = model(data_batch)
        _, output_batch = torch.max(output,1)
        # loss and accuracy
        batch_loss = loss_function(output, labels_batch)
        batch_accuracy = get_accuracy(output_batch,labels_batch)
        # update parameters
        model.zero_grad()
        batch_loss.backward()
        optimizer.step()
        # update loss and accuracy meter
        accuracy.update(batch_accuracy)
        loss.update(batch_loss.item())

    return accuracy.avg, loss.avg
        
        
        
        
        
        
        
