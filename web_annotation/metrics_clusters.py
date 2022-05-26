import pickle
import numpy as np
import pandas as pd
import networkx as nx

from tqdm import tqdm 

import sys
sys.path.insert(0, "../model/")
from utils import get_train_test_split, make_tree_orig, catch, find_most_similar_no_theo 


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
    existing_clusters = before['new_cluster'].unique()
    after = updated_morph[updated_morph['cluster_file'].str.contains(cluster_file)]
    additions = after[after['new_cluster'].isin(existing_clusters)]
    new_clusters = after[~after['new_cluster'].isin(existing_clusters)]

    scores = {
        'original size': before.shape[0],
        'newly added': after.shape[0],
        'additions to existing clusters': additions.shape[0],
        'number of clusters with new elements': additions['new_cluster'].nunique(),
        'new clusters' : new_clusters['new_cluster'].nunique(),
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
    positives = get_new_split(metadata,positives, morpho_graph_complete)

    positives = positives.groupby('uid_connection').first().reset_index()
    positive = positives.groupby('uid').last().reset_index()
    positive['old_cluster old'] = positive['cluster']
    positive['cluster'] = positive['new_cluster']
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


def get_new_split(metadata, positives, morpho_update):

    morpho_update = morpho_update[morpho_update["type"] == "POSITIVE"]
    morpho_update.columns = ["uid_connection", "img1", "img2", "type", "annotated", "cluster_file"]

    # creating connected components
    G = nx.from_pandas_edgelist(
        morpho_update,
        source="img1",
        target="img2",
        create_using=nx.DiGraph(),
        edge_key="uid_connection",
    )
    components = [x for x in nx.weakly_connected_components(G)]
    
    print(morpho_update.shape)

    # merging the two files
    positive = pd.concat(
        [
            positives,
            metadata.merge(morpho_update, left_on="uid", right_on="img1", how="inner"),
            metadata.merge(morpho_update, left_on="uid", right_on="img2", how="inner"),
        ],
        axis=0,
    ).groupby('uid_connection').first().reset_index()

    # adding set specification to df
    mapper = {it: number for number, nodes in enumerate(components) for it in nodes}
    positive["new_cluster"] = positive["uid"].apply(lambda x: mapper[x])

    positive['set'] = positive['set'].fillna('no set')

    old2new = {idx:set for idx, set in zip(positive.groupby('new_cluster')['set'].max().index, positive.groupby('new_cluster')['set'].max().values)}
    positive['new set'] = positive['new_cluster'].apply(lambda x: old2new[x])
    positive.loc[positive['new set'] == 'no set', 'new set'] = 'train'

    return positive

def make_new_train_set(embeddings, train_test, updated_morph, cluster_file, uid2path, data_dir='../data/'):
    'for each positive train with negative, if no negative, take closest one'
    after = updated_morph[updated_morph['cluster_file'].str.contains(cluster_file)]
    print(after.shape)
    
    tree, reverse_map = make_tree_orig(embeddings, reverse_map=True)
    Cs = []
    for i in tqdm(range(train_test.shape[0])):
        if after.loc[after['img1'] == train_test["uid"][i], :].loc[after['type'] == 'NEGATIVE', :].shape[0] > 0:
            list_sim = list(after.loc[after['img1'] == train_test["uid"][i], :].loc[after['type'] == 'NEGATIVE', 'img2'].values)
        else:
            list_theo = (
                list(train_test[train_test["img1"] == train_test["uid"][i]]["img2"])
                + list(train_test[train_test["img2"] == train_test["uid"][i]]["img1"])
                + [train_test["uid"][i]]
            )
            list_sim = find_most_similar_no_theo(
                train_test["uid"][i], tree, embeddings, reverse_map, list_theo, n=3
            )
            
        Cs.append(list_sim)
        
    train_test['C'] = Cs

    final = train_test[['img1', 'img2', 'C', 'set']].explode('C')
    final.columns = ['A', 'B', 'C', 'set']
    final['A_path'] = final['A'].apply(lambda x: catch(x, uid2path))
    final['B_path'] = final['B'].apply(lambda x: catch(x, uid2path))
    final['C_path'] = final['C'].apply(lambda x: catch(x, uid2path))
    print(final.shape)

    final = final[final['C_path'].notnull() & final['A_path'].notnull() & final['B_path'].notnull()]#.sample(frac=0.5)
    print(final.shape)
    print(final.tail())

    #final[final['set'] == 'train'].reset_index().to_csv(data_dir + 'dataset/abc_train_' + str(cluster_file.split('/')[-1]) + '.csv')
    return final[final['set'] == 'train'] 


def track_cluster_progression():
    return 'check if negatives were correctly pushed away'