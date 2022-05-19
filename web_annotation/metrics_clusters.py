import pickle
import numpy as np
import pandas as pd
import networkx as nx

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


def get_splits(morphograph):

    positives = morphograph[morphograph["type"] == "POSITIVE"]
    positives.columns = ["uid_connection", "img1", "img2", "type", "annotated", "cluster_file"]

    print(positives['cluster_file'].value_counts())
    # creating connected components
    G = nx.from_pandas_edgelist(
        positives,
        source="img1",
        target="img2",
        create_using=nx.DiGraph(),
        edge_key="uid_connection",
    )
    components = [x for x in nx.weakly_connected_components(G)]

    # adding set specification to df
    mapper = {it: number for number, nodes in enumerate(components) for it in nodes}
    positives["cluster"] = positives["img1"].apply(lambda x: mapper[x])
    positives["uid"] = positives["img1"]
    positives["set"] = [
        "test" if cl % 3 == 0 else "train" for cl in positives["cluster"]
    ]

    positives["set"] = [
        "val" if cl % 6 == 0 else set_ for cl, set_ in zip(positives["cluster"], positives["set"])
    ]

    return positives


def update_morph(data_dir):
    with open(data_dir + 'save_link_data_2018_08_02.pkl', 'rb') as f:
        morpho_graph_complete = pickle.load(f)
    morpho_graph_complete['cluster_file'] = 'Original'
    
    morpho_graph_clusters = pd.read_csv(data_dir + 'morphograph_clusters.csv')
    morpho_graph_clusters = morpho_graph_clusters.groupby(['img1', 'img2', 'type']).first().reset_index()
    morpho_graph_clusters.to_csv(data_dir + 'morphograph_clusters.csv', index=False)
    morpho_graph_clusters['uid'] = morpho_graph_clusters['uid_connection']
    morpho_graph_clusters['annotated'] = morpho_graph_clusters['date']
    morpho_graph_clusters = morpho_graph_clusters.drop(columns=['uid_connection', 'date', 'cluster'])
    morpho_graph_complete = pd.concat([morpho_graph_complete, morpho_graph_clusters], axis=0)
    
    
    positives = get_splits(morpho_graph_complete)
    print(positives.shape)

    positives = positives.groupby('uid_connection').first().reset_index()
    positive = positives.groupby('uid').last().reset_index()
    positive.to_csv(data_dir + 'morphograph/morpho_dataset.csv')
    
    return positive

