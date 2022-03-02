from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from utils import * 
import pickle


class ReplicaDataset(Dataset):
    """Replica dataset."""

    def __init__(self, csv_file, embeds_file, root_dir, transform=True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations. Path to train.csv or test.csv
            root_dir (string): Directory with all the images. Path to train or test folder
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = pd.read_csv(csv_file)

        with open(embeds_file, 'rb') as infile:
            self.embeds = pickle.load(infile)
  
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_1 = os.path.join(self.root_dir,
                             self.data.loc[idx, 'uid'] +
                             '.jpg')
        uid = self.data.loc[idx, 'uid']
        
        list_b = set(list(self.data[self.data['img1'] == uid]['img2']) + 
                     list(self.data[self.data['img2'] == uid]['img1']))
        
        img_2 = os.path.join(self.root_dir, list(list_b)[0] +
                             '.jpg')

        tree = make_tree(self.data, embeds=self.embeds, set_name=list(self.data['set'])[0])
        c = find_most_similar(self.data[self.data.index == idx], self.data, tree, embeds=self.embeds)
        img_3 = os.path.join(self.root_dir, c + '.jpg')

        if self.transform:
            A = preprocess_image(img_1)
            B = preprocess_image(img_2)
            C = preprocess_image(img_3)

        else:
            A = Image.open(img_1).unsqueeze(0)
            B = Image.open(img_2).unsqueeze(0)
            C = Image.open(img_3).unsqueeze(0)
        

        sample = {'A': A, 'B': B, 'C' : C}

        return sample
