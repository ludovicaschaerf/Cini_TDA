#!/usr/bin/python
# imports
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torch.nn.functional as F

import time
import copy
from tqdm import tqdm
import pandas as pd
import numpy as np

from utils import get_scores, get_embedding_test, make_training_set_orig
from custom_loss import TripletMarginWithDistanceLossCustom
from dataloader_replica import ReplicaDataset
from model_replica import ReplicaNet

from glob import glob
import pickle
import argparse
from datetime import datetime

# useful to log the run
now = datetime.now()
now = now.strftime("%d-%m-%Y_%H:%M:%S")


def main(data_dir, batch_size, num_epochs, model_name, device, resolution, num_c, retrain, continue_train, effort, create_embs):
    """Main function to train the triplet learning model. 

    Args:
        data_dir (Path or str): directory where metadata is stored
        batch_size (int): batch size
        num_epochs (int): number of epochs
        model_name (str): pretrained model name
        device (str): CPU or CUDA
        resolution (int): image resolution
        num_c (int): number of Cs retrieved for each tuple (A,B)
        retrain (bool): whether to start from the original data or from new step of annotations
        effort (int): number of retranining step
        continue_train (bool): whether to start from stratch or from previously stored weights
        predict (bool): whether to start from already stored embeddings or predict new ones

    Returns:
        _type_: _description_
    """
    # load utils file
    with open(data_dir + 'uid2path.pkl', 'rb') as outfile:
        uid2path = pickle.load(outfile)

    if device == 'cuda':
        # cuda requires batch size of 4
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    else:
        device = 'cpu'

    # instantiating the model, loss, optimizer and scheduler
    model = ReplicaNet(model_name, device)

    triplet_loss = TripletMarginWithDistanceLossCustom(
        distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y), margin=0.01,
        beta=0.1, reduction="sum", swap=True, intra=True
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-6, weight_decay=1e-4
    )

    scheduler = lr_scheduler.StepLR(
        optimizer, step_size=1000, gamma=0.01
    )

    if continue_train:  # loading previously stored weights
        if data_dir + "models/model_weights_" + now + model_name in glob(data_dir + "models/*"):
            print("loaded from previously stored weights")
            model.load_state_dict(torch.load(
                data_dir + "models/model_weights_" + now + model_name))

    if create_embs:
        # loading metadata file
        data = pd.read_csv(data_dir + "metadata.csv")
        # selecting only annotated images
        train_test = data[data["set"].notnull()].reset_index()
        
        # obtaining the embeddings for all the images
        embeddings = [[uid, get_embedding_test(path_file, model, device, resolution)] for uid, path_file in tqdm(
            zip(data['uid'].unique(), data['path'].unique()))]
        embeddings = np.array(embeddings, dtype=np.ndarray)
        
        # store for later use
        np.save(data_dir + 'embeddings/' + model_name +
                '_' + now + '.npy', embeddings)

        return get_scores(embeddings, train_test, data)

    else:
        # load already stored embeddings
        embeddings = np.load(
            data_dir + "embeddings/" + model_name + "_epoch_" + now + ".npy",
            allow_pickle=True,
        )

    if retrain:
        # loading the retrain dataset
        datasets = {x: ReplicaDataset(data_dir + 'dataset/retrain_'+str(
            effort)+'_' + x + '.csv', data_dir, x, resolution) for x in ['train', 'val']}

        data = pd.read_csv(data_dir + "data_retrain_" +
                           str(effort)+".csv")
        data['path'] = data['uid'].apply(lambda x: uid2path[x])

        train_test = data[data["set"].notnull()].reset_index()

        model = train_replica_f(
            model, datasets, data, train_test, triplet_loss, optimizer, scheduler, embeddings, uid2path,
            batch_size=batch_size, num_epochs=num_epochs, num_c=num_c, resolution=resolution, device=device,
            retrain=retrain, model_name=model_name,  data_dir=data_dir
        )

        torch.save(model.state_dict(), data_dir +
                   "models/model_weights_retrain_" + now + model_name + str(effort))

    else:
        data = pd.read_csv(data_dir + "data_sample.csv")
        train_test = data[data["set"].notnull()].reset_index()

        make_training_set_orig(embeddings, train_test,
                               data, data_dir, uid2path, epoch=10, n=num_c)
        
        print(pd.read_csv(data_dir + "dataset/abc_train_" +
              str(10) + ".csv")['C'].nunique())

        datasets = {x: ReplicaDataset(
            data_dir + 'dataset/abc_' + x + '_10.csv', data_dir, x, resolution) for x in ['train', 'val']}

        model = train_replica_f(
            model, datasets, data, train_test, triplet_loss, optimizer, scheduler, embeddings, uid2path,
            batch_size=batch_size, num_epochs=num_epochs, num_c=num_c, resolution=resolution, device=device,
            retrain=retrain, model_name=model_name,  data_dir=data_dir
        )
        
        # store model weights for future use
        torch.save(model.state_dict(), data_dir +
                   "models/model_weights_" + now + model_name)


def train_replica_f(
    model, datasets, data, train_test, triplet_loss, optimizer, scheduler, embeddings, uid2path,
    batch_size=4, num_epochs=5, num_c=5, resolution=320, device='cuda',
    retrain=False, model_name='resnext-101',  data_dir='/scratch/students/schaerf'
):
    """Function that carries out the training.

    Args:
        model: model
        datasets: train,val,test datasets
        triplet_loss: custom loss
        optimizer: optimizer
        scheduler: scheduler
        data (pd.DataFrame): information on all images
        train_test (pd.DataFrame): information on annotated images
        embeddings (np.array): embeddings of all images
        uid2path (dict): dict UIDs to path to images
        batch_size (int, optional): batch size. Defaults to 4.
        num_epochs (int, optional): number of epoch. Defaults to 5.
        num_c (int, optional): number of Cs. Defaults to 5.
        resolution (int, optional): image resolution. Defaults to 320.
        device (str, optional): CPU or CUDA. Defaults to 'cuda'.
        retrain (bool, optional): retrain step. Defaults to False.
        model_name (str, optional): model name (for storing). Defaults to 'resnext-101'.
        data_dir (str, optional): directory where data is stored. Defaults to '/scratch/students/schaerf'.

    Returns:
        Trained model.
    """

    dataset_sizes = {x: len(datasets[x]) for x in ["train", "val"]}

    # init dataset
    train_dataloaders = {
        x: DataLoader(datasets[x], batch_size=batch_size, shuffle=True)
        for x in ["train", "val"]
    }

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())

    best_loss = 100
    losses = []
    scores = []

    # freeze batch norm layers
    for param in model.modules():
        if isinstance(param, nn.BatchNorm2d):
            param.requires_grad = False

    # compute scores before training 
    scores.append(get_scores(embeddings, train_test, data))

    for epoch in range(num_epochs):

        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:  

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
                embeddings = [[uid, get_embedding_test(path_, model, device, resolution), ]
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

            # If model improve recreate the training sets
            if phase == "val" and (epoch_loss < best_loss or scores[-1][-1] > old_score) and not retrain:
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
                    x: DataLoader(
                        datasets[x], batch_size=batch_size, shuffle=True)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model specifics')
    parser.add_argument('--data_dir', dest='data_dir', type=str,
                        help='Directory where data is stored', default='/scratch/students/schaerf/')
    parser.add_argument('--batch_size', dest='batch_size',
                        type=int, help='Batch size', default=4)
    parser.add_argument('--num_epochs', dest='num_epochs',
                        type=int, help='Number of epochs', default=5)
    parser.add_argument('--num_c', dest='num_c', type=int,
                        help='Number of c', default=10)
    parser.add_argument('--resolution', dest='resolution',
                        type=int, help='Image resolution', default=320)
    parser.add_argument('--device', dest='device', type=str,
                        help='Device to use for computation', default='cuda')
    parser.add_argument('--model_name', dest='model_name', type=str,
                        help='Name of pretrained model to use', default='resnext-101')
    parser.add_argument('--retrain', dest='retrain', type=bool, default=False)
    parser.add_argument('--continue_train',
                        dest='continue_train', type=bool, default=False)
    parser.add_argument('--effort', dest='effort', type=int, default=1)
    parser.add_argument('--predict', dest='predict', type=bool, default=False)

    args = parser.parse_args()
    main(data_dir=args.data_dir, batch_size=args.batch_size, num_epochs=args.num_epochs,
         model_name=args.model_name, device=args.device, resolution=args.resolution,
         num_c=args.num_c, retrain=args.retrain, continue_train=args.continue_train,
         effort=args.effort, predict=args.predict)
