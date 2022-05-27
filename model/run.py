#!/usr/bin/python
from dataloader_replica import ReplicaDataset
from train_replica import train_replica, retrain_model
from model_replica import ReplicaNet

from glob import glob

import torch
#torch.cuda.empty_cache()
import gc
#gc.collect()

import pickle

import argparse
from datetime import datetime

now = datetime.now()
now = now.strftime("%d-%m-%Y_%H:%M:%S")
print(now)

def main(data_dir='/scratch/students/schaerf/', batch_size=8, num_epochs=1, model_name='resnext-101', device='cuda', resolution=480, num_c=10, retrain=False):
    dts = {x: ReplicaDataset(data_dir + 'dataset/abc_' + x + '.csv', data_dir, x, resolution) for x in ['train', 'val']}
    dataset_sizes = {x: len(dts[x]) for x in ["train", "val"]}

    with open(data_dir + 'uid2path.pkl', 'rb') as outfile:
        uid2path = pickle.load(outfile)

    if device == 'cuda':
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") # cuda requires batch size of 4
    else:
        device='cpu'
    
    model = ReplicaNet(model_name, device)

    noww = '25-05-2022_23:44:19' #'24-05-2022_22:50:41'#'24-05-2022_10:05:12' #'23-05-2022_17:14:25'#'30-04-2022_14:32:33'#'14-04-2022_23:25:30'
    if data_dir + "models/model_weights_" + noww + model_name in glob(data_dir + "models/*"):
       print("loaded from previously stored weights")
       model.load_state_dict(torch.load(data_dir + "models/model_weights_" + noww + model_name))

    if retrain:
        model = retrain_model(model, dts, dataset_sizes, device=device, effort='1', data_dir=data_dir, model_name=model_name,
                             resolution=resolution, batch_size=batch_size)
        torch.save(model.state_dict(), data_dir + "models/model_weights_retrain_" + now + model_name)

    else:
        model = train_replica(model, dts, dataset_sizes, uid2path, device=device, data_dir=data_dir, num_epochs=num_epochs, num_c=num_c, model_name=model_name, resolution=resolution, batch_size=batch_size)
        torch.save(model.state_dict(), data_dir + "models/model_weights_" + now + model_name)



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
    main(data_dir=args.data_dir, batch_size=args.batch_size, num_epochs=args.num_epochs, model_name=args.model_name, device=args.device, resolution=args.resolution, num_c=args.num_c, retrain=args.retrain)