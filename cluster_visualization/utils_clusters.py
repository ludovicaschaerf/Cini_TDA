import requests
from io import BytesIO
from PIL import Image
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN

def images_in_clusters(cluster_df, data):
    data_agg = {}
    for cluster in cluster_df['cluster'].unique():
        if cluster not in [-1]:
            data_agg[cluster] = []
            rows = cluster_df[cluster_df['cluster'] == cluster]
            for row in rows.iterrows():
                row_2 = data[data['uid'] == row[1]['uid']]
                info_2 = str(row_2["AuthorOriginal"].values[0]) + '\n ' + str(row_2["Description"].values[0])
        
                drawer = row[1]['path'].split('/')[0]
                img = row[1]['path'].split('_')[1].split('.')[0]
                image = f'https://dhlabsrv4.epfl.ch/iiif_replica/cini%2F{drawer}%2F{drawer}_{img}.jpg/full/300,/0/default.jpg'
                data_agg[cluster].append([info_2,image])
    return data_agg
    

def draw_clusters(cluster_num, cluster_df, data):
    rows = cluster_df[cluster_df['cluster'] == cluster_num]
    n = rows.shape[0]
    
    f, axarr = plt.subplots((n // 4 + 1),4, figsize=(30,7* n // 4 + 1))
    
    plt.suptitle('Cluster n '+ str(cluster_num))
    axarr = axarr.flatten()
    for i,row in enumerate(rows.iterrows()):
        row_2 = data[data['uid'] == row[1]['uid']]
        info_2 = str(row_2["AuthorOriginal"].values[0]) + '\n ' + str(row_2["Description"].values[0])
    
        axarr[i].set_title(info_2)
        drawer = row[1]['path'].split('/')[0]
        img = row[1]['path'].split('_')[1].split('.')[0]
        image = requests.get(f'https://dhlabsrv4.epfl.ch/iiif_replica/cini%2F{drawer}%2F{drawer}_{img}.jpg/full/300,/0/default.jpg')
        
        axarr[i].imshow(Image.open(BytesIO(image.content))) #replica_dir + 
        
    plt.savefig('../figures/cluster_' + str(cluster_num) + '.jpg')
    plt.show()
    
def setup(data_dir = '../data/'):
    with open(data_dir + 'uid2path.pkl', 'rb') as outfile:
        uid2path = pickle.load(outfile)
    with open(data_dir + 'list_iconography.pkl', 'rb') as infile:
        final = pickle.load(infile)
    return data_dir, uid2path, final

def make_clusters(data_dir):
    data_dir, uid2path, final = setup(data_dir)
    sim_mat = np.load(data_dir + 'similarities_madonnas_2600.npy', allow_pickle=True) #embedding_no_pool/)
    diff_mat = np.round(1 - sim_mat, 3)
    db = DBSCAN(eps=0.03, min_samples=2, metric='precomputed').fit(diff_mat)
    labels = final[:2600]
    classes = db.labels_

    clusters = pd.DataFrame({'uid':labels, 'cluster':classes})
    #print(clusters['cluster'].value_counts(), clusters['cluster'].nunique())
    clusters['path'] = clusters['uid'].apply(lambda x: uid2path[x])
    return clusters
