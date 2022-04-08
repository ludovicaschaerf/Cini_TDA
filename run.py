#!/usr/bin/python
from dataloader_replica import ReplicaDataset
from train_replica import train_replica
from model_replica import ReplicaNet
from glob import glob
import torch
torch.cuda.empty_cache()
import gc
gc.collect()
import pickle

import argparse
from datetime import datetime

now = datetime.now()
now = now.strftime("%d-%m-%Y_%H:%M:%S")
print(now)

def main(data_dir='/scratch/students/schaerf/', replica_dir='/mnt/project_replica/datasets/cini/', batch_size=8, num_epochs=1, model_name='resnext-101', device='cuda', resolution=480):
    dts = {x: ReplicaDataset(data_dir + 'dataset/abc_' + x + '.csv', data_dir + 'subset.csv', replica_dir, data_dir, x, resolution) for x in ['train', 'val']}
    dataset_sizes = {x: len(dts[x]) for x in ["train", "val"]}

    with open(data_dir + 'uid2path.pkl', 'rb') as outfile:
        uid2path = pickle.load(outfile)

    if device == 'cuda':
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") # cuda requires batch size of 4
    else:
        device='cpu'
    
    model = ReplicaNet(model_name, device)

    #if data_dir + "models/model_weights_" + model_name in glob(data_dir + "models/*"):
    #    print("loaded from previously stored weights")
    #    model.load_state_dict(torch.load(data_dir + "models/model_weights_" + now + model_name))

    model = train_replica(model, dts, dataset_sizes, uid2path, device=device, data_dir=data_dir, replica_dir=replica_dir, num_epochs=num_epochs, model_name=model_name, resolution=resolution, batch_size=batch_size)
    torch.save(model.state_dict(), data_dir + "models/model_weights_" + now + model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model specifics')
    parser.add_argument('--data_dir', dest='data_dir', type=str, help='Directory where data is stored', default='/scratch/students/schaerf/')
    parser.add_argument('--replica_dir', dest='replica_dir', type=str, help='Directory where data is stored', default='/mnt/project_replica/datasets/cini/')
    parser.add_argument('--batch_size', dest='batch_size', type=int, help='Batch size', default=4)
    parser.add_argument('--num_epochs', dest='num_epochs', type=int, help='Number of epochs', default=1)
    parser.add_argument('--resolution', dest='resolution', type=int, help='Image resolution', default=320)
    parser.add_argument('--device', dest='device', type=str, help='Device to use for computation', default='cuda')
    parser.add_argument('--model_name', dest='model_name', type=str, help='Name of pretrained model to use', default='resnext-101')

    args = parser.parse_args()
    main(args.data_dir, args.replica_dir, args.batch_size, args.num_epochs, args.model_name, args.device, args.resolution)