import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torch.nn.functional as F

import time
from datetime import datetime
import copy
from tqdm import tqdm
import pandas as pd

# from scipy import sparse

from utils import *


now = datetime.now()
now = now.strftime("%d-%m-%Y_%H:%M:%S")


def train_replica(
    model,
    loaders,
    dataset_sizes,
    uid2path,
    device="cpu",
    data_dir="/scratch/students/schaerf/",
    replica_dir="/mnt/project_replica/datasets/cini/",
    model_name="resnext-101",
    resolution=360,
    num_epochs=20,
    batch_size=8,
):

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())

    #triplet_loss = nn.TripletMarginLoss(
     #   margin=0.0001, reduction="mean"  # to be optimized margin
    #)

    triplet_loss = nn.TripletMarginWithDistanceLoss(
        distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y), margin=0.01, reduction="sum"
    )

    # optimizer = torch.optim.Adagrad(model.parameters(), lr=1e-6) # to be optimized lr and method
    
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-6
    )  # to be optimized lr and method
    scheduler = lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.01
    )  # to be optimized step and gamma

    best_loss = 10000000
    losses = []
    scores = []

    data = pd.read_csv(data_dir + "dedup_data_sample.csv").drop(
        columns=["Unnamed: 0", "level_0"]
    )

    #embeddings = [[uid, get_embedding(preprocess_image(replica_dir + path, resolution=resolution), model, device=device)] for uid, path in tqdm(zip(data['uid'].unique(), data['path'].unique()))]
    #embeddings = np.array(embeddings, dtype=np.ndarray)
    #np.save(data_dir + 'embeddings/' + model_name + '_epoch_none' + now + '.npy', embeddings)

    noww = '14-04-2022_23:25:29' #'14-04-2022_08:27:32' #"06-04-2022_09:33:39"  #'04-04-2022_19:55:56'
    embeddings = np.load(
        data_dir + "embeddings/" + model_name + "_epoch_3" + noww + ".npy",
        allow_pickle=True,
    )

    train_test = data[data["set"].notnull()].reset_index()

    scores.append(get_scores(embeddings, train_test, data))

    make_training_set_orig(embeddings, train_test, data, data_dir, uid2path, epoch=10, n=20)
    loaders["train"].__reload__(data_dir + "dataset/abc_train_" + str(10) + ".csv")
    loaders["val"].__reload__(data_dir + "dataset/abc_val_" + str(10) + ".csv")
    train_dataloaders = {
        x: DataLoader(loaders[x], batch_size=batch_size, shuffle=True)
        for x in ["train", "val"]
    }

    for param in model.modules():
        if isinstance(param, nn.BatchNorm2d):
            param.requires_grad = False


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
                embeddings = [
                    [
                        uid,
                        get_embedding(
                            preprocess_image(replica_dir + path, resolution=resolution),
                            model,
                            device=device,
                        ),
                    ]
                    for uid, path in tqdm(
                        zip(data["uid"].unique(), data["path"].unique())
                    )
                ]
                embeddings = np.array(embeddings, dtype=np.ndarray)
                np.save(
                    data_dir
                    + "embeddings/"
                    + model_name
                    + "_epoch_"
                    + str(epoch)
                    + now
                    + ".npy",
                    embeddings,
                )

                old_score = scores[-1][-1]
                scores.append(get_scores(embeddings, train_test, data))
                print('boh', scores[-1][-1])

            # deep copy the model
            if phase == "val" and (epoch_loss < best_loss or scores[-1][-1] < old_score):  # needs to be changed to val
                print("Model updating! Best loss so far")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

                make_training_set_orig(
                    embeddings, train_test, data, data_dir, uid2path, epoch=epoch
                )
                loaders["train"].__reload__(
                    data_dir + "dataset/abc_train_" + str(epoch) + ".csv"
                )
                loaders["val"].__reload__(
                    data_dir + "dataset/abc_val_" + str(epoch) + ".csv"
                )
                train_dataloaders = {
                    x: DataLoader(loaders[x], batch_size=batch_size, shuffle=True)
                    for x in ["train", "val"]
                }

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val loss: {:4f}".format(best_loss))
    # print("Best val acc: {:4f}".format(max(accuracies)))

    # load best model weights
    model.load_state_dict(best_model_wts)

    pd.DataFrame(
        scores,
        columns=[
            "mean_position",
            "mean_min_position",
            "mean_median_position",
            "map",
            "recall_400",
            "recall_200",
            "recall_100",
            "recall_50",
            "recall_20",
        ],
    ).to_csv(data_dir + "scores/scores_" + str(now) + ".csv")

    return model
