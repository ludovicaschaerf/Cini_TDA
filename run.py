#!/usr/bin/python
from dataloader_replica import ReplicaDataset
from train_replica import train_replica
from model_replica import ReplicaNet
from torch.utils.data import DataLoader
from glob import glob
import torch
import argparse

def main(data_dir='/scratch/students/schaerf/', batch_size=8, num_epochs=1):
    dts = {x: ReplicaDataset(data_dir + x + '.csv', data_dir + 'dict2emb.pkl', data_dir + 'subset.csv', data_dir, x) for x in ['train', 'test']}
    train_dataloaders = {x: DataLoader(dts[x], batch_size=batch_size, shuffle=True) for x in ["train", "test"]}
    dataset_sizes = {x: len(dts[x]) for x in ["train", "test"]}

    model = ReplicaNet()

    if data_dir + "model_weights" in glob(data_dir + "model_weights"):
        print("loaded from previously stored weights")
        model.load_state_dict(torch.load(data_dir + "model_weights"))

    model = train_replica(model, train_dataloaders, dataset_sizes, dts, num_epochs=num_epochs)
    torch.save(model.state_dict(), data_dir + "model_weights")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model specifics')
    parser.add_argument('--data_dir', dest='data_dir', type=str, help='Directory where data is stored', default='/scratch/students/schaerf/')
    parser.add_argument('--batch_size', dest='batch_size', type=int, help='Batch size', default=8)
    parser.add_argument('--num_epochs', dest='num_epochs', type=int, help='Number of epochs', default=1)

    args = parser.parse_args()
    main(args.data_dir, args.batch_size, args.num_epochs)



