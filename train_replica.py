import torch
from torch import nn
import time
import copy
from torch.optim import lr_scheduler
from tqdm import tqdm
import pickle
from scipy import sparse
import pandas as pd
from utils import make_training_set
from torch.utils.data import DataLoader


def train_replica(model, loaders, dataset_sizes, device='cpu', data_dir='/scratch/students/schaerf/', num_epochs=20, batch_size=8):

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())

    triplet_loss = nn.TripletMarginLoss(
        margin=0.01, reduction='sum' # to be optimized margin
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5) # to be optimized lr and method
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1) # to be optimized step and gamma

    best_loss = 1000
    #train_accuracy = 0
    #test_accuracy = 0
    #accuracies = []
    losses = []
    
    train_dataloaders = {x: DataLoader(loaders[x], batch_size=batch_size, shuffle=True) for x in ["train", "test"]}
    
    for epoch in range(num_epochs):
               
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "test"]:  # , 'val'
            
            running_loss = 0.0

            # Iterate over data.
            for a, b, c in tqdm(train_dataloaders[phase]):
                a = a.squeeze(1).to(device)
                b = b.squeeze(1).to(device)
                c = c.squeeze(1).to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):

                    for param in model.modules():
                        if isinstance(param, nn.BatchNorm2d):
                            param.requires_grad = False
                    
                    # Forward pass
                    A, B, C = model(a, b, c)
                    # Compute and print loss
                    loss = triplet_loss(A, B, C)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * a.size(0)
                
                

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            if phase == 'test':
                losses.append(epoch_loss)

            print("{} Loss: {:.4f}".format(phase, epoch_loss))

            
            # deep copy the model
            if (
                phase == "test" and epoch_loss < best_loss
            ):  # needs to be changed to val
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
        
        
        if epoch % 10 == 0:
            subset = pd.read_csv(data_dir + 'subset.csv')
            make_training_set(data_dir + 'subset/', model, subset)
            loaders.__reload__(data_dir + 'abc_train.csv')
            loaders.__reload__(data_dir + 'abc_test.csv')
            train_dataloaders = {x: DataLoader(loaders[x], batch_size=batch_size, shuffle=True) for x in ["train", "test"]}
    


    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val loss: {:4f}".format(best_loss))
    #print("Best val acc: {:4f}".format(max(accuracies)))

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model
