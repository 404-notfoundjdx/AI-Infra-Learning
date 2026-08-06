'''
Contains functionality for creating Pytorch DataLoader's for image classification data
'''
import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = 1
def create_dataloader(train_dir:str,
                      test_dir:str,
                      transform:transforms.Compose,
                      batch_size:int,
                      num_works:int=NUM_WORKERS):
    train_data = datasets.ImageFolder(root=train_dir,
                                      transform=transform,
                                      target_transform=None)

    test_data = datasets.ImageFolder(root=test_dir,
                                     transform=transform)

    class_names = train_data.classes

    train_dataloader = DataLoader(train_data,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_works,
                                  pin_memory=True)
    
    test_dataloader = DataLoader(test_data,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_works,
                                 pin_memory=True)

    return train_dataloader, test_dataloader, class_names
