"""
Contains the functionality for creating PyTorch DataLoaders for image classification data.
"""

import os

import torch
import torchvision
from torchvision import datasets, transforms 
from torch.utils.data import DataLoader 

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    train_dir: str,
    test_dir: str,
    train_transform: transforms.Compose,
    test_transform: transforms.Compose,
    batch_size: int,
    num_workers: int=NUM_WORKERS
    ):
    
    """ Creates Training and Testing DataLoaders.
    
    Takes in a training directory and testing directory path and 
    turns them into PyTorch Datasets and then into PyTorch DataLoaders.
    
    Args:
        train_dir: Path to training directory.
        test_dir: Path to testing directory.
        transform: torchvision transforms to perform on training and testing data.
        batch_size: number of samples per batch in each of the DataLoaders.
        num_workers: An integer for number of workers per DataLoader 
    
    Returns:
        A tuple of (train_dataloader, test_dataloader , class_names).
        where class_names is a list of the target classes.
    
    Example Usage:
        train_dataloader, test_dataloader, class_names = create_dataLoader(
                                                                           train_dir = path/to/train_dir,
                                                                           test_dir = path/to/test_dir,
                                                                           train_transform = some_transform,
                                                                           test_tansform = some_transform,
                                                                           batch_size = 32,
                                                                           num_workers = 2
                                                                           )
    """
    
    # Use ImageFolder to create Dataset(s)
    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    test_data = datasets.ImageFolder(test_dir, transform=test_transform)
    
    # Get class names :
    class_names = train_data.classes
    
    # Turn image into DataLoaders 
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
        )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,               # Don't need to shuffle test data
        num_workers=num_workers,
        pin_memory=True
        )
        
    
    return train_dataloader, test_dataloader, class_names 
