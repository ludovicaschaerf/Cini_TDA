#!/usr/bin/python
import argparse
from datetime import datetime
import pickle
import pandas as pd
import numpy as np
import sys

sys.path.insert(0, "../model/")

from utils import show_suggestions, make_tree_orig, find_most_similar_no_theo, catch

def setup(data_dir='/scratch/students/schaerf/', path='/home/guhennec/scratch/2021_Cini/TopologicalAnalysis_Cini/data/', size=1000):
    data = pd.read_csv(data_dir + 'dedup_data.csv').drop(columns=['Unnamed: 0', 'level_0']).sample(size) #'full_data.csv').drop(columns=['Unnamed: 0', 'level_0'])
    #embeddings = np.load(path + 'Replica_UIDs_ResNet_VGG_All.npy',allow_pickle=True)
    embeddings = np.load(data_dir + 'resnext-101_epoch_901-05-2022_19%3A45%3A03.npy',allow_pickle=True)
    embeddings = embeddings[np.isin(embeddings[:,0], list(data["uid"].unique()))]
    tree, reverse_map = make_tree_orig(embeddings, reverse_map=True)


    with open(data_dir + 'uid2path_old.pkl', 'rb') as outfile:
        uid2path = pickle.load(outfile)

    return embeddings, data, tree, reverse_map, uid2path


def get_links(embeddings, data, tree, reverse_map, uid2path, uid=False, n=8):

    train_test = data[data["set"].notnull()].reset_index() 
    if uid:
        row = data[data['uid'] == uid]
    else:
        row = data.sample()
    
    if row["set"].values[0] in ['train', 'test']:
        list_theo = (
            list(train_test[train_test["img1"] == row["uid"].values[0]]["img2"])
            + list(train_test[train_test["img2"] == row["uid"].values[0]]["img1"])
            + [row["uid"].values[0]]
        )
    else:
        list_theo = [row["uid"].values[0]]
        
    sim = find_most_similar_no_theo(
        row["uid"].values[0], tree, embeddings, reverse_map, list_theo, n=n
    )

    images = []
    drawer = row["path"].values[0].split('/')[0]
    img = row["path"].values[0].split('_')[1].split('.')[0]
    info = row["AuthorOriginal"].values[0] + ' ' + row["Description"].values[0]
    image_a = f'https://dhlabsrv4.epfl.ch/iiif_replica/cini%2F{drawer}%2F{drawer}_{img}.jpg/full/300,/0/default.jpg'
        
    for i in range(len(sim)):
        drawer = catch(sim[i], uid2path).split('/')[0]
        img = catch(sim[i], uid2path).split('_')[1].split('.')[0]
        row_2 = data[data['uid'] == sim[i]]
        info_2 = row_2["AuthorOriginal"].values[0] + ' ' + row_2["Description"].values[0]
    
        images.append((sim[i], f'https://dhlabsrv4.epfl.ch/iiif_replica/cini%2F{drawer}%2F{drawer}_{img}.jpg/full/400,/0/default.jpg', info_2))
        
    
    return (row["uid"].values[0], image_a, info), images

    

def store_morph(uid_a, uid_sim, data_dir='/scratch/students/schaerf/annotation/'):
    with open(data_dir + 'morphograph_update.pkl', 'rb') as f:
        morpho_complete  = pickle.load(f)
    now = datetime.now()
    now = now.strftime("%d-%m-%Y_%H:%M:%S")
    
    
    new_morphs = pd.DataFrame([[uid_a[:16]+uid_sim[i][16:], uid_a, uid_sim[i], 'POSITIVE', now] for i in range(len(uid_sim))], columns=['uid', 'img1', 'img2', 'type', 'annotated'])
    update = pd.concat([morpho_complete, new_morphs], axis=0)
    print(update.tail())
    print(morpho_complete.shape, update.shape)
    with open(data_dir + 'morphograph_update.pkl', 'wb') as f:
        pickle.dump(update, f)
    

def main(
    data_dir="./data/",
    embeddings=False,
    data=False,
    embds=False,
    tree=False,
    reverse_map=False,
    uid2path=False,
    size=1000,
):

    now = datetime.now()
    now = now.strftime("%d-%m-%Y_%H:%M:%S")

    if not embds:
        embeddings, data, tree, reverse_map, uid2path = setup(data_dir, size) #np.load(data_dir + 'embeddings/benoit.npy',allow_pickle=True)
    
    train_test = data[data["set"].notnull()].reset_index() 
    a, sim = show_suggestions(data.sample(), embeddings, train_test, tree, reverse_map, uid2path, data)
    similars = input('which ones are VERY similar but NOT equal in pose? \n Write the numbers separated by commas')
    
    uids_sim = [sim[int(i.strip())] for i in similars.split(',') if i != '']
    store_morph(a, uids_sim)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model specifics")
    parser.add_argument(
        "--data_dir",
        dest="data_dir",
        type=str,
        help="Directory where data is stored",
        default="./data/",
    )
    parser.add_argument(
        "--embeddings",
        dest="embeddings",
        type=str,
        help="Which embeddings to use",
        default="benoit",
    )

    args = parser.parse_args()
    main(args.data_dir, args.embeddings)
