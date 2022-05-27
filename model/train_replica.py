#!/usr/bin/python

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torch.nn.functional as F

from custom_loss import TripletMarginWithDistanceLossCustom
import time
import copy
from tqdm import tqdm
import pandas as pd

from utils import *

from dataloader_replica import ReplicaDataset
from model_replica import ReplicaNet

from glob import glob
import pickle

import argparse
from datetime import datetime

now = datetime.now()
now = now.strftime("%d-%m-%Y_%H:%M:%S")
print(now)

def main(data_dir, batch_size, num_epochs, model_name, device, resolution, num_c, retrain):
    if device == 'cuda':
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") # cuda requires batch size of 4
    else:
        device='cpu'
    
    model = ReplicaNet(model_name, device)
    
    noww = '25-05-2022_23:44:19'

    if data_dir + "models/model_weights_" + noww + model_name in glob(data_dir + "models/*"):
       print("loaded from previously stored weights")
       model.load_state_dict(torch.load(data_dir + "models/model_weights_" + noww + model_name))

    with open(data_dir + 'uid2path.pkl', 'rb') as outfile:
        uid2path = pickle.load(outfile)

    triplet_loss = TripletMarginWithDistanceLossCustom(
        distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y), margin=0.01, 
        beta=0.13, reduction="sum", swap=True, intra=True
    )
    
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-6
    )  
    
    scheduler = lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.01
    )  
    
    
    # embeddings = [[uid, catch_error(path_, model, device, resolution)] for uid, path_ in tqdm(zip(data['uid'].unique(), data['path'].unique()))]
    # embeddings = np.array(embeddings, dtype=np.ndarray)
    # np.save(data_dir + 'embeddings/' + model_name + '_epoch_none' + now + '.npy', embeddings)

    embeddings = np.load(
        data_dir + "embeddings/" + model_name + "_epoch_22" + noww + ".npy",
        allow_pickle=True,
    )

    if retrain:
        datasets = {x: ReplicaDataset(data_dir + 'dataset/retrain_1_'  + x + '.csv', data_dir, x, resolution) for x in ['train', 'val']}
        
        data = pd.read_csv(data_dir + "data_retrain_1.csv").drop(columns=['level_0'])
        train_test = data[data["set"].notnull()].reset_index()

        model = train_replica_f(
            model, datasets, data, train_test, triplet_loss, optimizer, scheduler, embeddings, uid2path,
            batch_size=batch_size, num_epochs=num_epochs, num_c=num_c, resolution=resolution, device=device, 
            retrain=retrain, model_name=model_name,  data_dir=data_dir
        )
        
        torch.save(model.state_dict(), data_dir + "models/model_weights_retrain_" + now + model_name)

    else:
        data = pd.read_csv(data_dir + "data_sample.csv")
        train_test = data[data["set"].notnull()].reset_index()

        make_training_set_orig(embeddings, train_test, data, data_dir, uid2path, epoch=10, n=num_c)
        print(pd.read_csv(data_dir + "dataset/abc_train_" + str(10) + ".csv")['C'].nunique())
        
        datasets = {x: ReplicaDataset(data_dir + 'dataset/abc_'+ x +'_10.csv', data_dir, x, resolution) for x in ['train', 'val']}
        
        model = train_replica_f(
            model, datasets, data, train_test, triplet_loss, optimizer, scheduler, embeddings, uid2path,
            batch_size=batch_size, num_epochs=num_epochs, num_c=num_c, resolution=resolution, device=device, 
            retrain=retrain, model_name=model_name,  data_dir=data_dir
        )
        torch.save(model.state_dict(), data_dir + "models/model_weights_" + now + model_name)

def train_replica_f(
    model, datasets, data, train_test, triplet_loss, optimizer, scheduler, embeddings, uid2path,
    batch_size=4, num_epochs=5, num_c=5, resolution=320, device='cuda', 
    retrain=False, model_name='resnext-101',  data_dir='/scratch/students/schaerf'
):

    dataset_sizes = {x: len(datasets[x]) for x in ["train", "val"]}
        
    train_dataloaders = {
        x: DataLoader(datasets[x], batch_size=batch_size, shuffle=True)
        for x in ["train"]
    }
        
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    
    best_loss = 100
    losses = []
    scores = []

    for param in model.modules():
        if isinstance(param, nn.BatchNorm2d):
            param.requires_grad = False

    scores.append(get_scores(embeddings, train_test, data))

    for epoch in range(num_epochs):

        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:  # , 'val'
            
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

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

            epoch_loss = running_loss * batch_size / dataset_sizes[phase]
            losses.append(epoch_loss)

            print("{} Loss: {:.4f}".format(phase, epoch_loss))

            if phase == "val":
                embeddings = [[uid, catch_error(path_, model, device, resolution),]
                    for uid, path_ in tqdm(zip(data["uid"].unique(), data["path"].unique()))
                ]
                embeddings = np.array(embeddings, dtype=np.ndarray)
                np.save(
                    data_dir + "embeddings/" + model_name + "_epoch_"
                    + str(epoch) + now + ".npy", embeddings,
                )
                print(embeddings.shape)
                old_score = scores[-1][-1]
                scores.append(get_scores(embeddings, train_test, data))
                
            # deep copy the model
            if phase == "val" and (epoch_loss < best_loss or scores[-1][-1] > old_score) and not retrain:  # needs to be changed to val
                print("Model updating! Best loss so far")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

                make_training_set_orig(
                    embeddings, train_test, data, data_dir, uid2path, epoch=epoch, n=num_c
                )
                datasets["train"].__reload__(
                    data_dir + "dataset/abc_train_" + str(epoch) + ".csv"
                )
                datasets["val"].__reload__(
                    data_dir + "dataset/abc_val_" + str(epoch) + ".csv"
                )
                train_dataloaders = {
                    x: DataLoader(datasets[x], batch_size=batch_size, shuffle=True)
                    for x in ["train", "val"]
                }
            

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val loss: {:4f}".format(best_loss))
    
    # load best model weights
    model.load_state_dict(best_model_wts)

    pd.DataFrame(
        scores,
        columns=[
            "mean_position", "mean_min_position",
            "mean_median_position", "map", "recall_400",
            "recall_200", "recall_100", "recall_50", "recall_20",
        ],
    ).to_csv(data_dir + "scores/scores_" + str(now) + ".csv")

    return model


def catch_error(path, model, device, resolution):
    try:
        return get_embedding(preprocess_image_test(path, resolution=resolution), model, device=device)
    except Exception as e:
        print(e)
        return np.zeros(2048)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model specifics')
    parser.add_argument('--data_dir', dest='data_dir', type=str, help='Directory where data is stored', default='/scratch/students/schaerf/')
    parser.add_argument('--batch_size', dest='batch_size', type=int, help='Batch size', default=4)
    parser.add_argument('--num_epochs', dest='num_epochs', type=int, help='Number of epochs', default=5)
    parser.add_argument('--num_c', dest='num_c', type=int, help='Number of c', default=10)
    parser.add_argument('--resolution', dest='resolution', type=int, help='Image resolution', default=320)
    parser.add_argument('--device', dest='device', type=str, help='Device to use for computation', default='cuda')
    parser.add_argument('--model_name', dest='model_name', type=str, help='Name of pretrained model to use', default='resnext-101')
    parser.add_argument('--retrain', dest='retrain', type=bool, default=False)
    args = parser.parse_args()
    main(data_dir=args.data_dir, batch_size=args.batch_size, num_epochs=args.num_epochs, 
         model_name=args.model_name, device=args.device, resolution=args.resolution, 
         num_c=args.num_c, retrain=args.retrain)
    
    





#'24-05-2022_22:50:41'#'24-05-2022_10:05:12'#'23-05-2022_17:14:25' #'19-05-2022_10:33:39' 
#'13-05-2022_14:35:30' #'30-04-2022_14:32:33' #'29-04-2022_23:38:51' #'29-04-2022_17:29:42' 
#'14-04-2022_08:27:32' #"06-04-2022_09:33:39"  #'04-04-2022_19:55:56' '14-04-2022_23:25:29' 
    