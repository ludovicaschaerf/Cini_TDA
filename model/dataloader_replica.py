from __future__ import print_function, division
import os

import torch
from torch.utils.data import Dataset

import pandas as pd

from utils import *


class ReplicaDataset(Dataset):
    """Replica dataset loader. """

    def __init__(self, csv_file, root_dir, phase, resolution=480):
        """
        Args:
            csv_file (string): Path to the csv file with annotations. Path to train.csv or test.csv
            root_dir (string): Directory with all the images. Path to train or test folder
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = pd.read_csv(csv_file)
        self.resolution = resolution
        self.root_dir = root_dir
        self.phase = phase
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        A = preprocess_image(self.data.loc[idx, "A_path"], resolution=self.resolution)
        B = preprocess_image(self.data.loc[idx, "B_path"], resolution=self.resolution)
        C = preprocess_image(self.data.loc[idx, "C_path"], resolution=self.resolution)

        sample = [A, B, C]

        return sample

    def __show_images__(self, idx):
        show_images([self.data.loc[idx, "A_path"], 
                     self.data.loc[idx, "B_path"], 
                     self.data.loc[idx, "C_path"]])

    
    def __reload__(self, csv_file):
        #del self.data
        self.data = pd.read_csv(csv_file)
        print('reloaded data', self.data.shape)
        