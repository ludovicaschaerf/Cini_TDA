import torch
from torch import nn
import time
import copy
from torch.optim import lr_scheduler
from tqdm import tqdm
import pickle
from scipy import sparse
import pandas as pd
from utils import *
from torch.utils.data import DataLoader
from store_embeddings import *

def train_replica(model, loaders, dataset_sizes, device='cpu', data_dir='/scratch/students/schaerf/', model_name='resnext-101', resolution=360, num_epochs=20, batch_size=8):

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())

    triplet_loss = nn.TripletMarginLoss(
        margin=0.01, reduction='sum' # to be optimized margin
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5) # to be optimized lr and method
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.01) # to be optimized step and gamma

    best_loss = 1000
    losses = []
    scores = []
    
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
            losses.append(epoch_loss)

            print("{} Loss: {:.4f}".format(phase, epoch_loss))

            
            # deep copy the model
            if (
                phase == "test" and epoch_loss < best_loss
            ):  # needs to be changed to val
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
        
        
                if epoch % 2 == 0:
                    data = pd.read_csv(data_dir + 'full_data.csv').drop(columns=['Unnamed: 0', 'level_0'])
                    embeddings = [[uid, get_embedding(preprocess_image(data_dir + 'subset/' + uid + '.jpg', resolution=resolution), model, device=device).squeeze(1).squeeze(1)] for uid in tqdm(data['uid'].unique())]
                    embeddings = np.array(embeddings, dtype=np.ndarray)
                    np.save(data_dir + model_name + '_epoch_' + str(epoch) + '.npy', embeddings)
            
                    list_downloaded = [
                        file.split("/")[-1].split(".")[0] for file in glob(data_dir + "subset/*")
                    ]
                    print(len(list_downloaded))

                    train_test = data[data["set"].notnull()].reset_index() 
                    print(train_test.shape)

                    scores.append(get_scores(embeddings, train_test, data, list_downloaded))
            
                    make_training_set_orig(embeddings, train_test, data, data_dir, epoch=epoch)
                    loaders[phase].__reload__(data_dir + 'abc_train_' + epoch + '.csv')
                    loaders[phase].__reload__(data_dir + 'abc_test' + epoch + '.csv')
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
    
    pd.Series(losses, name='loss').to_csv(data_dir + 'losses.csv')
    pd.DataFrame(scores, columns=['mean_position', 'mean_min_position', 'mean_median_position', 'map', 'recall_400', 'recall_200', 'recall_100', 'recall_50', 'recall_20']).to_csv(data_dir + 'scores.csv')
    
    return model
