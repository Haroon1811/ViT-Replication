"""
Creates functions for training and testing a PyTorch model.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_step(model: nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_func: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    
    """
    Trains a PyTorch model for a single epoch.
    
    Turns a target PyTorch model to training mode and then runs 
    through all of the required training steps(forward pass, loss calculation, optimization)
    
    Args:
        model: A PyTorch model to be trained.
        dataloader: A DataLoader instance for the model to be trained on.
        loss_func: A PyTorch loss function to minimize.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        device: A target device to compute on.
        
    Returns: 
        A tuple of training loss and training accuracy metrics.
        In the form(train_loss, train_accuracy)
        
    """
    # Put model in train mode 
    model.train()
    
    # Setup loss and accuracy values 
    train_loss, train_accuracy = 0, 0
    
    # Loop through data loader batches
    for batch, (data, label) in enumerate(dataloader):
        # send data to target device
        data, label = data.to(device), label.to(device)
        
        #1. Forward pass
        y_logit = model(data)
        
        #2. loss calculation and accumulation
        loss = loss_func(y_logit, label)
        train_loss += loss.item()
        
        #3. Optimizer zero grad
        optimizer.zero_grad()
        
        #4. Loss backwards (Backpropagation)
        loss.backward()
        
        #5. Optimizer step (Gradient descent)
        optimizer.step()
        
        #6. calculate and accumulate accuracy metric across all batches 
        y_pred_class = torch.argmax(torch.softmax(y_logit, dim=1), dim=1)
        train_accuracy += (y_pred_class == label).sum().item()/len(y_logit)
    
    # Adjust metrics to get average loss and accuracy
    train_loss = train_loss / len(dataloader)
    train_accuracy =  train_accuracy / len(dataloader)
    
    return train_loss, train_accuracy
        
    
    

    
def test_step(model: nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_func: torch.nn.Module,
               device: torch.device) -> Tuple[float, float]:
    
    """
    Tests a PyTorch model for a single epoch.
    
    Turns a target PyTorch model to 'eval' mode and then runs 
    through the forward pass, loss calculation on a testing dataset.
        
    Args:
        model: A PyTorch model to be tested.
        dataloader: A DataLoader instance for the model to be tested on.
        loss_func: A PyTorch loss function to minimize.
        device: A target device to compute on.
        
    Returns: 
        A tuple of testing loss and testing accuracy metrics.
        In the form(test_loss, test_accuracy)
        
    """
    # Put model in train mode 
    model.eval()
    
    # Setup loss and accuracy values 
    test_loss, test_accuracy = 0, 0
    
    # Turn on the inference context manager 
    with torch.inference_mode():
        
        # Loop through data loader batches
        for batch, (data, label) in enumerate(dataloader):
            # send data to target device
            data, label = data.to(device), label.to(device)

            #1. Forward pass
            y_logit = model(data)

            #2. loss calculation and accumulation
            loss = loss_func(y_logit, label)
            test_loss += loss.item()

            #3. calculate and accumulate accuracy metric across all batches 
            y_pred_class = torch.argmax(torch.softmax(y_logit, dim=1), dim=1)
            test_accuracy += (y_pred_class == label).sum().item()/len(y_logit)

        # Adjust metrics to get average loss and accuracy
        test_loss = test_loss / len(dataloader)
        test_accuracy =  test_accuracy / len(dataloader)

        return test_loss, test_accuracy
    
    

def train(model: nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          loss_func: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          epochs: int,
          device: torch.device) -> Dict[str, List]:
    
    """
        Trains and tests a PyTorch model.
        
        Passes a target PyTorch models through train_step() and test_step()
        functions for a number of epochs, training and testing the model in the same epoch loop
        
        Calculates, prints and stores evaluation metrics throughout.
        
        Args:
        
        model: A PyTorch model to be trained.
        train_dataloader: A DataLoader instance for the model to be trained on.
        test_dataloader: A DataLoader instance for the model to be tested on.
        loss_func: A PyTorch loss function to minimize loss on both datasets.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        epochs: An integer indicating how many epochs to train on.
        device: A target device to compute on.
        
        Returns:
        
        A dictionary of training and testing loss as well as training 
        and testing accuracy metrics. Each metric has a value in a list for each epoch.
        In the form: {train_loss: [...],
                      train_accuracy: [...],
                      test_loss: [...],
                      test_loss: [...]
                      }
    """
    # Create an empty dictionary 
    results = {"train_loss": [],
               "train_accuracy": [],
               "test_loss": [],
               "test_accuracy": []
              }
    
    # Loop through training and testing steps for number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_accuracy = train_step(model=model,
                                                dataloader=train_dataloader,
                                                loss_func=loss_func,
                                                optimizer=optimizer,
                                                device=device)
        test_loss, test_accuracy = test_step(model=model,
                                             dataloader=test_dataloader,
                                             loss_func=loss_func,
                                             device=device)
        # print out what's happening 
        print(f"Epoch: {epoch+1} |"
              f"Training loss: {train_loss:.4f} |"
              f"Training accuracy: {train_accuracy:.4f} |"
              f"Testing loss: {test_loss:.4f} |"
              f"Testing accuracy: {test_accuracy:.4f} |"
             )
        
        # update the results dictionary
        results["train_loss"].append(train_loss)
        results["train_accuracy"].append(train_accuracy)
        results["test_loss"].append(test_loss)
        results["test_accuracy"].append(test_accuracy)
    
    return results
        
