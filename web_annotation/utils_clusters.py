import requests
from io import BytesIO
from PIL import Image
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from glob import glob

from sklearn.cluster import DBSCAN

def images_in_clusters(cluster_df, data):
    data_agg = {}
    for cluster in cluster_df['cluster'].unique():
        if cluster not in [-1]:
            data_agg[cluster] = []
            rows = cluster_df[cluster_df['cluster'] == cluster]
            for row in rows.iterrows():
                row_2 = data[data['uid'] == row[1]['uid']]
                info_2 = str(row_2["AuthorOriginal"].values[0]) + '\n ' + str(row_2["Description"].values[0]) + '\n ' + str(row_2["BeginDate"].values[0]) + '\n ' + str(row_2["set"].values[0])
                uid = row[1]['uid']
                    
                if 'cini' in row[1]['path']:
                    drawer = row[1]['path'].split('/')[-2]
                    img = row[1]['path'].split('_')[1].split('.')[0]
                    image = f'https://dhlabsrv4.epfl.ch/iiif_replica/cini%2F{drawer}%2F{drawer}_{img}.jpg/full/300,/0/default.jpg'
                elif 'WGA' in row[1]['path']:
                    drawer = '/'.join(row[1]['path'].split('/')[-3]).split('.')[0] # http://www.wga.hu/html/a/aachen/allegory.html
                    image = f'http://www.wga.hu/html/{drawer}.html'
                else:
                    print(row[1]['path'])  
                    continue
                      
                data_agg[cluster].append([info_2,image, uid])
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
        #'http://www.wga.hu/html/a/aachen/allegory.html'
        axarr[i].imshow(Image.open(BytesIO(image.content))) #replica_dir + 
        
    plt.savefig('../figures/cluster_' + str(cluster_num) + '.jpg')
    plt.show()
    
def setup_clusters(data_dir = '../data/'):
    with open(data_dir + 'uid2path.pkl', 'rb') as outfile:
        uid2path = pickle.load(outfile)
    with open(data_dir + 'list_iconography.pkl', 'rb') as infile:
        final = pickle.load(infile)
    return data_dir, uid2path, final

def make_clusters(data_dir):
    data_dir, uid2path, final = setup_clusters(data_dir)
    sim_mat = np.load(data_dir + 'similarities_madonnas_2600.npy', allow_pickle=True) #embedding_no_pool/)
    diff_mat = np.round(1 - sim_mat, 3)
    db = DBSCAN(eps=0.03, min_samples=2, metric='precomputed').fit(diff_mat)
    labels = final[:2600]
    classes = db.labels_

    clusters = pd.DataFrame({'uid':labels, 'cluster':classes})
    #print(clusters['cluster'].value_counts(), clusters['cluster'].nunique())
    clusters['path'] = clusters['uid'].apply(lambda x: uid2path[x])
    return clusters



def make_clusters_embeddings(data_dir, dist=0.5):
    data = pd.read_csv(data_dir + 'dedup_data_sample_wga.csv')
    embeds = np.load(data_dir + 'resnext-101_epoch_901-05-2022_19%3A45%3A03.npy', allow_pickle=True) #embedding_no_pool/)
    
    uids = list(data['uid'])
    
    uid2path = {}
    for i, row in data.iterrows():
        uid2path[row['uid']] = row['path']
    
    db = DBSCAN(eps=dist, min_samples=2, metric='euclidean').fit(np.vstack(embeds[:,1])) #0.52 best so far
    labels = embeds[:,0]
    classes = db.labels_

    clusters = pd.DataFrame({'uid':labels, 'cluster':classes})
    print(clusters.shape)
    
    print(clusters['cluster'].value_counts(), clusters['cluster'].nunique())
    
    clusters = clusters[clusters['uid'].isin(uids)].reset_index()
    print(clusters.shape)
    clusters['path'] = clusters['uid'].apply(lambda x: uid2path[x])
    print(clusters.shape)
    return clusters


def store_morph_cluster(imges_uids_sim, info_cluster, cluster_num, cluster_file, data_dir='/scratch/students/schaerf/annotation/'):
    morpho = pd.read_csv(data_dir + 'morphograph_clusters.csv')
    
    now = datetime.now()
    now = now.strftime("%d-%m-%Y_%H:%M:%S")
    
    to_add = []
    
    for uid in imges_uids_sim:
        for info in info_cluster:
            if uid == info[2] and info[0][-3:] == 'nan':
                for uid2 in imges_uids_sim:
                    if uid2 != uid:
                        to_add.append([uid[:16]+uid2[16:], uid, uid2, 'POSITIVE', now, cluster_file, cluster_num])

    new_morphs = pd.DataFrame(to_add, columns=['uid_connection', 'img1', 'img2', 'type', 'date', 'cluster_file', 'cluster'])
    update = pd.concat([morpho, new_morphs], axis=0)
    print(update.tail())
    print(morpho.shape, update.shape)
    update.to_csv(data_dir + 'morphograph_clusters.csv', index=False)

def store_morph_cluster_negatives(imges_uids_sim, info_cluster, cluster_num, cluster_file, data_dir='/scratch/students/schaerf/annotation/'):
    morpho = pd.read_csv(data_dir + 'morphograph_clusters.csv')
    

    now = datetime.now()
    now = now.strftime("%d-%m-%Y_%H:%M:%S")
    
    to_add = []
    
    for uid in imges_uids_sim:
        for info in info_cluster:
            if uid == info[2] and info[0][-3:] == 'nan':
                for uid2 in imges_uids_sim:
                    if uid2 != uid:
                        to_add.append([uid[:16]+uid2[16:], uid, uid2, 'POSITIVE', now, cluster_file, cluster_num])

    different_images = [info[2] for info in info_cluster if info[2] not in imges_uids_sim] 
    for uid in imges_uids_sim:
        for uid2 in different_images:
            to_add.append([uid[:16]+uid2[16:], uid, uid2, 'NEGATIVE', now, cluster_file, cluster_num])

    new_morphs = pd.DataFrame(to_add, columns=['uid_connection', 'img1', 'img2', 'type', 'date', 'cluster_file', 'cluster'])
    update = pd.concat([morpho, new_morphs], axis=0)
    print(update[['img1', 'img2', 'type', 'cluster']].tail())
    print(morpho.shape, update.shape)
    update.to_csv(data_dir + 'morphograph_clusters.csv', index=False)
