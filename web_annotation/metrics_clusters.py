import pickle
import numpy as np
import pandas as pd
import networkx as nx

import sys
sys.path.insert(0, "../model/")
from utils import get_train_test_split


def cluster_accuracy(cluster_annotations):
    cluster_info = cluster_annotations.groupby('cluster')['type'].apply(lambda x: x.value_counts()).reset_index()
    scores = {}
    for cluster, group in cluster_info.groupby('cluster'):
        if group.shape[0] == 1:
            if group['level_1'].values[0] in ['POSITIVE', 'CORRECT']:
                scores[cluster] = 1
            elif group['level_1'].values[0] in ['WRONG', 'NEGATIVE']:
                scores[cluster] = 0
        elif 'POSITIVE' in group['level_1'].values:
            scores[cluster] = group[group['level_1'] == 'POSITIVE'].shape[0] / group.shape[0]
        
    return np.round(np.mean(list(scores.values())), 2)


def novelty_score(updated_morph, cluster_file, previous_cluster='Original'):
    before = updated_morph[updated_morph['cluster_file'] == previous_cluster]
    existing_clusters = before['cluster'].unique()
    after = updated_morph[updated_morph['cluster_file'].str.contains(cluster_file)]
    additions = after[after['cluster'].isin(existing_clusters)]
    new_clusters = after[~after['cluster'].isin(existing_clusters)]

    scores = {
        'original size': before.shape[0],
        'newly added': after.shape[0],
        'additions to existing clusters': additions.shape[0],
        'number of clusters with new elements': additions.cluster.nunique(),
        'new clusters' : new_clusters.cluster.nunique(),
        'new clusters elements': new_clusters.shape[0],
        'progress': str(np.round(after.shape[0] / before.shape[0] * 100, 2)) + '%'

    }
    return scores


def update_morph(data_dir):
    with open(data_dir + 'save_link_data_2018_08_02.pkl', 'rb') as f:
        morpho_graph_complete = pickle.load(f)
    morpho_graph_complete['cluster_file'] = 'Original'
    
    metadata = pd.read_csv(data_dir + 'data_sample.csv')
    metadata = metadata.drop(columns=['img1', 'img2', 'type', 'annotated', 'index', 'cluster', 'set', 'uid_connection'])

    ## function take what was already in train and test and preserve it (make train test split and then add the new ones)
    positives = get_train_test_split(metadata, morpho_graph_complete)
    
    morpho_graph_clusters = pd.read_csv(data_dir + 'morphograph_clusters.csv')
    morpho_graph_clusters = morpho_graph_clusters.groupby(['img1', 'img2', 'type']).first().reset_index()
    morpho_graph_clusters.to_csv(data_dir + 'morphograph_clusters.csv', index=False)
    morpho_graph_clusters['uid'] = morpho_graph_clusters['uid_connection']
    morpho_graph_clusters['annotated'] = morpho_graph_clusters['date']
    morpho_graph_clusters = morpho_graph_clusters.drop(columns=['uid_connection', 'date', 'cluster'])
    morpho_graph_complete = pd.concat([morpho_graph_complete, morpho_graph_clusters], axis=0)
    
    print(positives.shape)

    positives = positives.groupby('uid_connection').first().reset_index()
    positive = positives.groupby('uid').last().reset_index()
    positive.to_csv(data_dir + 'morphograph/morpho_dataset.csv')
    
    return positive


def evaluate_morph(updated_morph, original_cluster='Original'):
    morph_original = updated_morph[updated_morph['cluster_file'] == original_cluster]
    scores = {
        'precision': 'each cluster with a morpho how many it catches that should be together / size of cluster',
        'recall' : 'how many it catches per cluster / how many there are to catch',
        'accuracy' : ''
    }
    return scores


def make_new_train_set():
    return 'for each positive train with negative, if no negative, take closest one'


def track_cluster_progression():
    return 'check if negatives were correctly pushed away'