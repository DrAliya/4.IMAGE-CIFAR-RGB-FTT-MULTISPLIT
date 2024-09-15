from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy





def evaluate_model(model,dataloader,dataset_sizes,device):
    model.eval()
    with torch.no_grad():
        running_corrects=0
        total=0
        sum_diff_time=0
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
    accuracy = running_corrects.double() /dataset_sizes


    return accuracy





   
def simple_train(model, criterion, optimizer, scheduler, 
                dataloaders, dataset_sizes, device,num_epochs=25, model_state_path=None):
    
    
    '''
    train and at the same time save the weights in a dictionary
    '''

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
#     print("going to save init weight")
#     weight_description=store_weights_in_dic(weight_description,model)    
#     weight_description=store_weights_in_dic_conti(weight_description,model)    
    

    
    for epoch in range(num_epochs):
        
        start_epoch=time.time()
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        
        
        # do the training part
        model.train()  # Set model to training mode
        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for inputs, labels in dataloaders["train"]:
            
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()      
                
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
            
        scheduler.step()
        epoch_loss = running_loss / dataset_sizes["train"]
        epoch_acc = running_corrects.double() / dataset_sizes["train"]
        print(f'Training Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        

        
        
        
        # go for validation now
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in dataloaders["val"]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                                          
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)   
                                          
        epoch_loss = running_loss / dataset_sizes["val"]
        epoch_acc = running_corrects.double() / dataset_sizes["val"]

        print(f'val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        if epoch_acc > best_acc:
            print(epoch_acc,"is better than",best_acc)
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            if model_state_path:
                print("Saving")
                torch.save(model.state_dict(), model_state_path)

        end_epoch=time.time()
        print("Time for an epoch train and val = ",(end_epoch-start_epoch)/60,"minutes")
    time_elapsed = time.time() - since
    print(f'Training {num_epochs} complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    model.load_state_dict(best_model_wts)
    
    return model

