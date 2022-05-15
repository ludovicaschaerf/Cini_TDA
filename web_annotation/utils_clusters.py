from flask import Flask, render_template, request
import requests
from io import BytesIO
from PIL import Image
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import json
from glob import glob

from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

def catch(x, uid2path):
    try:
        return uid2path[x]
    except:
        return [0,0]


def annotate_store(INFO, cluster_file, data_dir):
    if request.method == "POST":
        if request.form["submit"]:
            imges_uids_sim = []
            for form_key in request.form.keys():
                if "ckb" in form_key:
                    imges_uids_sim.append(request.form[form_key])
            cluster_num = int(request.form["form"])
            
            if request.form["submit"] == "similar_images":
                store_morph_cluster(imges_uids_sim, INFO[int(request.form["form"])], cluster_num, cluster_file, data_dir=data_dir)

            if request.form["submit"] == "both_images":
                store_morph_cluster_negatives(imges_uids_sim, INFO[int(request.form["form"])], cluster_num, cluster_file, data_dir=data_dir)

            if request.form["submit"] == "wrong":
                store_wrong_cluster(INFO[int(request.form["form"])], cluster_num, cluster_file, data_dir=data_dir)


def images_in_clusters(cluster_df, data, data_dir='../data/', map_file='map2pos_10-05-2022.pkl'):
    data_agg = {}
    with open(data_dir + map_file, 'rb') as infile:
        map2pos = pickle.load(infile)

    if not 'annotated' in data.columns:
        data['annotated'] = ''
    
    cluster_df['pos'] = cluster_df['uid'].apply(lambda x: catch(x, map2pos))
    for cluster in cluster_df['cluster'].unique():
        if cluster not in [-1]:
            data_agg[int(cluster)] = []
            rows = cluster_df[cluster_df['cluster'] == cluster]
            for row in rows.iterrows():
                row_2 = data[data['uid'] == row[1]['uid']]
                info_2 = str(row_2["AuthorOriginal"].values[0]) + '\n ' + str(row_2["Description"].values[0]) + '\n ' + str(row_2["BeginDate"].values[0]) + '\n ' + str(row_2["annotated"].fillna('').values[0].split('_')[0].split(' ')[0]) + '\n ' + str(row_2["set"].values[0]) 
                uid = row[1]['uid']
                pos = row[1]['pos']    
                if 'ImageURL' in cluster_df.columns:
                        if 'html' in row[1]['ImageURL']: 
                            image = row[1]['ImageURL'].split('html')[0]+'art'+row[1]['ImageURL'].split('html')[1] +'jpg'
                        else:
                            image = row[1]['ImageURL']
                elif 'WGA' in row[1]['path']:
                    drawer = '/'.join(row[1]['path'].split('/')[6:]).split('.')[0] # http://www.wga.hu/html/a/aachen/allegory.html
                    image = f'http://www.wga.hu/art/{drawer}.jpg'
                else:
                    try:
                        drawer = row[1]['path'].split('/')[-1].split('_')[0]
                        img = row[1]['path'].split('/')[-1].split('_')[1].split('.')[0]
                        image = f'https://dhlabsrv4.epfl.ch/iiif_replica/cini%2F{drawer}%2F{drawer}_{img}.jpg/full/300,/0/default.jpg'
                    except:
                        image = ''
                      
                data_agg[int(cluster)].append([info_2, image, uid, float(pos[0]), float(pos[1])])
    return data_agg
    

    

def make_clusters(data_dir='../data/', uid2path_file = 'uid2path.pkl', final_file='list_iconography.pkl', embed_file='similarities_madonnas_2600.npy'):
    with open(data_dir + uid2path_file, 'rb') as outfile:
        uid2path = pickle.load(outfile)
    with open(data_dir + final_file, 'rb') as infile:
        final = pickle.load(infile)
    
    sim_mat = np.load(data_dir + embed_file, allow_pickle=True) #embedding_no_pool/)
    diff_mat = np.round(1 - sim_mat, 3)
    db = DBSCAN(eps=0.03, min_samples=2, metric='precomputed').fit(diff_mat)
    labels = final[:sim_mat.shape[0]]
    classes = db.labels_

    clusters = pd.DataFrame({'uid':labels, 'cluster':classes})
    #print(clusters['cluster'].value_counts(), clusters['cluster'].nunique())
    clusters['path'] = clusters['uid'].apply(lambda x: catch(x, uid2path))

    return clusters

def make_clusters_embeddings(data_dir='../data/', data_file='data_wga_cini_45000.csv', embed_file='resnext-101_epoch_410-05-2022_10%3A11%3A05.npy', dist=0.5, min_n=2, type_clustering='dbscan'):
    
    data = pd.read_csv(data_dir + data_file)
    embeds = np.load(data_dir + embed_file, allow_pickle=True) #embedding_no_pool/)
    uids = list(data['uid'])
    
    uid2path = {}
    for i, row in data.iterrows():
        uid2path[row['uid']] = row['path']
    
    if type_clustering=='dbscan':
        db = DBSCAN(eps=dist, min_samples=min_n, metric='euclidean').fit(np.vstack(embeds[:,1])) #0.52 best so far
        classes = db.labels_
    else:
        km = KMeans(n_clusters=dist, max_iter=100, n_init=10).fit(np.vstack(embeds[:,1]))
        classes = km.labels_

    labels = embeds[:,0]
        
    clusters = pd.DataFrame({'uid':labels, 'cluster':classes})
    print(clusters.shape)
    
    print(clusters['cluster'].value_counts(), clusters['cluster'].nunique())
    
    clusters = clusters[clusters['uid'].isin(uids)].reset_index()
    print(clusters.shape)
    clusters['path'] = clusters['uid'].apply(lambda x: uid2path[x])

    print(clusters.shape)
    return clusters

def make_links(data_hierarchical):
    cluster_lists = data_hierarchical.groupby('cluster_desc')['cluster'].apply(lambda x: list(x))
    pairs_to_match = []
    for list_ in cluster_lists:
        for i in range(len(list_)):
            for j in range(len(list_) - i):
                if list_[j] != list_[i]:
                    if list_[j] != -1 and list_[i] != -1:
                        pairs_to_match.append(list(set([str(list_[i]),str(list_[j])])))
    return list(set(['-'.join(pair) for pair in pairs_to_match]))


def convert_to_json(data_agg):
    new = ''
    for cluster in data_agg.keys():
        new  += '!!' + str(cluster) + '%%' + '%%'.join(['$$'.join([str(c) for c in cli]) for cli in data_agg[cluster]])
    return new


def get_2d_pos(data_dir='../data/', embed_file='resnext-101_epoch_410-05-2022_10%3A11%3A05.npy'):
    embeds = np.load(data_dir + embed_file, allow_pickle=True) #embedding_no_pool/)
    
    embeddings_new = TSNE(
            n_components=2
        ).fit_transform(np.vstack(embeds[:, 1]))
    map2pos = {}
    for i, uid in enumerate(embeds[:,0]):
        map2pos[uid] = embeddings_new[i]
    return map2pos

def store_wrong_cluster(info_cluster, cluster_num, cluster_file, data_dir='/scratch/students/schaerf/annotation/'):
    morpho = pd.read_csv(data_dir + 'morphograph_wrong_clusters.csv')

    now = datetime.now()
    now = now.strftime("%d-%m-%Y_%H:%M:%S")

    to_add = []
    for info in info_cluster:
        if info[0][-3:] != 'nan':
            for info_2 in info_cluster:
                if info_2[0][-3:] == 'nan':
                        to_add.append([info[2][:16]+info_2[2][16:], info[2], info_2[2], 'NEGATIVE', now, cluster_file, cluster_num])


    new_morphs = pd.DataFrame(to_add, columns=['uid_connection', 'img1', 'img2', 'type', 'date', 'cluster_file', 'cluster'])
    update = pd.concat([morpho, new_morphs], axis=0)
    print(update.tail())
    print(morpho.shape, update.shape)
    update.to_csv(data_dir + 'morphograph_wrong_clusters.csv', index=False)

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


### deprecated
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
