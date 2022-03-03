from dataloader_replica import ReplicaDataset
from train_replica import train_replica
from model_replica import ReplicaNet
from torch.utils.data import DataLoader

dt = ReplicaDataset('train.csv', 'dict2emb', 'train')
train_dataloaders = {x: DataLoader(dt, batch_size=8, shuffle=True) for x in ['train']}
dataset_sizes = {x: len(dt) for x in ['train']}

model = ReplicaNet()

train_replica(model, train_dataloaders, dataset_sizes, dt, num_epochs=2)