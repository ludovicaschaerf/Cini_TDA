from dataloader_replica import ReplicaDataset
from train_replica import train_replica
from model_replica import ReplicaNet
from torch.utils.data import DataLoader
from glob import glob
import torch

dt = ReplicaDataset("/scratch/students/schaerf/train.csv", "/scratch/students/schaerf/dict2emb", "/scratch/students/schaerf/train")
train_dataloaders = {x: DataLoader(dt, batch_size=8, shuffle=True) for x in ["train"]}
dataset_sizes = {x: len(dt) for x in ["train"]}

model = ReplicaNet()

if "model_weights" in glob("model_weights"):
    print("loaded from previously stored weights")
    model.load_state_dict(torch.load("model_weights"))

train_replica(model, train_dataloaders, dataset_sizes, dt, num_epochs=1)
