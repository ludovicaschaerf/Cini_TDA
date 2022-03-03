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

    def __init__(self, csv_file, embeds_file, root_dir):
        """
        Args:
            csv_file (string): Path to the csv file with annotations. Path to train.csv or test.csv
            root_dir (string): Directory with all the images. Path to train or test folder
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = pd.read_csv(csv_file).groupby('uid').agg({'img1':lambda x: list(x),'img2':lambda x: list(x), 'set':'first'}).reset_index()
        self.embeds_file = embeds_file
        self.root_dir = root_dir
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_1 = os.path.join(self.root_dir,
                             self.data.loc[idx, 'uid'] +
                             '.jpg')
        
        list_b = set(list(self.data.loc[idx,'img2']) + 
                     list(self.data.loc[idx,'img1']))
        
        img_2 = os.path.join(self.root_dir, list(list_b)[0] +
                             '.jpg')

        with open(self.embeds_file, 'rb') as infile:
            embeds = pickle.load(infile)
        
        tree = make_tree(self.data, embeds=embeds, set_name=list(self.data['set'])[0])
        c = find_most_similar(self.data[self.data.index == idx], self.data, tree, embeds=embeds)
        img_3 = os.path.join(self.root_dir, c + '.jpg')

        A = preprocess_image(img_1)
        B = preprocess_image(img_2)
        C = preprocess_image(img_3)


        sample = [A, B, C]

        return sample

    def __get_simgle_item__(self, idx):
        img_1 = os.path.join(self.root_dir,
                             self.data.loc[idx, 'uid'] +
                             '.jpg')
        uid = self.data.loc[idx, 'uid']
        A = preprocess_image(img_1)
        return uid, A