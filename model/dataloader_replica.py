from __future__ import print_function, division

import torch
from torch.utils.data import Dataset
import pandas as pd
from utils import preprocess_image, show_images


class ReplicaDataset(Dataset):
    """Replica dataset loader. """

    def __init__(self, csv_file, root_dir, resolution=480):
        """
        Initialising the variables. 
        Args:
            csv_file (string): Path to the csv file with annotations. 
            resolution (string): Image resolution inputted into the model.
            root_dir (string): Directory with all the images. Path to train or test folder

        """
        self.data = pd.read_csv(csv_file) #training data
        self.resolution = resolution
        self.root_dir = root_dir

    def __len__(self):
        """Returns length of the data being loaded.

        Returns:
            int: length of dataset. 
        """
        return len(self.data)

    def __getitem__(self, idx):
        """Loads the input tuple of A, B, C for triplet learning.

        Args:
            idx (int): loader index 

        Returns:
            _type_: tuple of (A,B,C)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        A = preprocess_image(
            self.data.loc[idx, "A_path"], resolution=self.resolution)
        B = preprocess_image(
            self.data.loc[idx, "B_path"], resolution=self.resolution)
        C = preprocess_image(
            self.data.loc[idx, "C_path"], resolution=self.resolution)

        sample = (A, B, C)

        return sample

    def __show_images__(self, idx):
        """Helper function to show the sampled triplet.

        Args:
            idx (int): loader index
        """
        show_images([self.data.loc[idx, "A_path"],
                     self.data.loc[idx, "B_path"],
                     self.data.loc[idx, "C_path"]])

    def __reload__(self, csv_file):
        """Helper function to reload the metadata with the input triplets 
           after the end of epoch update.

        Args:
            csv_file (posix.Path or str): full path to the metadata file.
        """
        self.data = pd.read_csv(csv_file)
        print('Data successfully reloaded! New data has shape:', self.data.shape)
