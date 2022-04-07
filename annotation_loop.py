#!/usr/bin/python
from glob import glob
import argparse
from datetime import datetime
import pickle
import pandas as pd
import numpy as np
from store_embeddings import show_suggestions, make_tree_orig

def get_data(data_dir='/scratch/students/schaerf/', size=1000):
    data = pd.read_csv(data_dir + 'dedup_data.csv').drop(columns=['Unnamed: 0', 'level_0']).sample(size) #'full_data.csv').drop(columns=['Unnamed: 0', 'level_0'])
    path = '/home/guhennec/scratch/2021_Cini/TopologicalAnalysis_Cini/data/'
    embeddings = np.load(path + 'Replica_UIDs_ResNet_VGG_All.npy',allow_pickle=True)
    print(embeddings.shape)
    embeddings = embeddings[np.isin(embeddings[:,0], list(data["uid"].unique()))]
    tree, reverse_map = make_tree_orig(embeddings, reverse_map=True)
    return embeddings, data, tree, reverse_map

def main(data_dir='/scratch/students/schaerf/', embeddings=False, data=False, embds=False, tree=False, reverse_map=False, size=1000):
    
    with open(data_dir + 'annotation/morphograph_update.pkl', 'rb') as f:
        morpho_complete  = pickle.load(f)
    now = datetime.now()
    now = now.strftime("%d-%m-%Y_%H:%M:%S")
    
    if not embds:
        embeddings, data, tree, reverse_map = get_data(data_dir, size) #np.load(data_dir + 'embeddings/benoit.npy',allow_pickle=True)
    
    train_test = data[data["set"].notnull()].reset_index() 
    uids = embeddings[:,0]
    
    # data[data['uid'] == uid]
    a, sim = show_suggestions(data.sample(), embeddings, train_test, tree, reverse_map)
    similars = input('which ones are VERY similar but NOT equal in pose? \n Write the numbers separated by commas')
    
    uids_sim = [sim[int(i.strip())] for i in similars.split(',') if i != '']
    
    morpho_graph = morpho_complete.loc[((morpho_complete.img1.isin(uids)) & (morpho_complete.img2.isin(uids))) ][['uid', 'img1', 'img2', 'type', 'annotated']]   
    new_morphs = pd.DataFrame([[a[:16]+uids_sim[i][16:], a, uids_sim[i], 'POSITIVE', now] for i in range(len(uids_sim))], columns=['uid', 'img1', 'img2', 'type', 'annotated'])
    update = pd.concat([morpho_graph, new_morphs], axis=0)
    print(update.tail())
    print(morpho_graph.shape, update.shape)
    with open(data_dir + 'annotation/morphograph_update.pkl', 'wb') as f:
        pickle.dump(update, f)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model specifics')
    parser.add_argument('--data_dir', dest='data_dir', type=str, help='Directory where data is stored', default='/scratch/students/schaerf/')
    parser.add_argument('--embeddings', dest='embeddings', type=str, help='Which embeddings to use', default='benoit')
    
    args = parser.parse_args()
    main(args.data_dir, args.embeddings)