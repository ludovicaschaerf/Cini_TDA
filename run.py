#!/usr/bin/python
from dataloader_replica import ReplicaDataset
from train_replica import train_replica
from model_replica import ReplicaNet
from torch.utils.data import DataLoader
from glob import glob
import torch
torch.cuda.empty_cache()
import gc
gc.collect()

import argparse

def main(data_dir='/scratch/students/schaerf/', batch_size=8, num_epochs=1, device='gpu'):
    dts = {x: ReplicaDataset(data_dir + x + '.csv', data_dir + 'dict2emb.pkl', data_dir + 'subset.csv', data_dir, x) for x in ['train', 'test']}
    train_dataloaders = {x: DataLoader(dts[x], batch_size=batch_size, shuffle=True) for x in ["train", "test"]}
    dataset_sizes = {x: len(dts[x]) for x in ["train", "test"]}

    if device == 'gpu':
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # cuda requires batch size of 4
    else:
        device='cpu'
    
    model = ReplicaNet(device)

    if data_dir + "model_weights" in glob(data_dir + "model_weights"):
        print("loaded from previously stored weights")
        #model.load_state_dict(torch.load(data_dir + "model_weights"))

    model = train_replica(model, train_dataloaders, dataset_sizes, dts, device=device, data_dir=data_dir, num_epochs=num_epochs, batch_size=batch_size)
    torch.save(model.state_dict(), data_dir + "model_weights")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model specifics')
    parser.add_argument('--data_dir', dest='data_dir', type=str, help='Directory where data is stored', default='/scratch/students/schaerf/')
    parser.add_argument('--batch_size', dest='batch_size', type=int, help='Batch size', default=8)
    parser.add_argument('--num_epochs', dest='num_epochs', type=int, help='Number of epochs', default=1)
    parser.add_argument('--device', dest='device', type=str, help='Device to use for computation', default='gpu')

    args = parser.parse_args()
    main(args.data_dir, args.batch_size, args.num_epochs, args.device)



