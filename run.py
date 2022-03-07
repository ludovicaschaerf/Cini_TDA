from dataloader_replica import ReplicaDataset
from train_replica import train_replica
from model_replica import ReplicaNet
from torch.utils.data import DataLoader
from glob import glob
import torch

data_dir = '/scratch/students/schaerf/'
dts = {x: ReplicaDataset(data_dir + x + ".csv", data_dir + "dict2emb", data_dir + x) for x in ["train", "test"]}
train_dataloaders = {x: DataLoader(dts[x], batch_size=8, shuffle=True) for x in ["train", "test"]}
dataset_sizes = {x: len(dts[x]) for x in ["train", "test"]}

model = ReplicaNet()

if data_dir + "model_weights" in glob(data_dir + "model_weights"):
    print("loaded from previously stored weights")
    model.load_state_dict(torch.load(data_dir + "model_weights"))

model = train_replica(model, train_dataloaders, dataset_sizes, dts, num_epochs=1)
torch.save(model.state_dict(), data_dir + "model_weights")
