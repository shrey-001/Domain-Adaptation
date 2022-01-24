import numpy as np
from .HGMDataset import HGM
from .transforms import transform_dummy,transform_train,transform_test
from torch.utils.data import DataLoader,RandomSampler

def get_dataloaders(
        annotations_file,
        train_dir,
        val_dir,
        train_transform=None,
        val_transform=None,
        batch_size=16,
        *args, **kwargs):
    """
    This function returns the train, val and test dataloaders.
    """
    train_dataset = HGM(annotations_file,train_dir,train_transform)
    val_dataset = HGM(annotations_file,val_dir,val_transform)

    train_dl = DataLoader(train_dataset,sampler=RandomSampler(train_dataset), batch_size=batch_size, drop_last=False,num_workers=8, *args, **kwargs)
    val_dl = DataLoader(val_dataset,sampler=RandomSampler(val_dataset), batch_size=batch_size, drop_last=False,num_workers=8, *args, **kwargs)

    return train_dl, val_dl