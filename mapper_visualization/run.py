print('\n> (0/6) Starting \n')
import numpy as np
import sklearn
import pandas as pd
import pickle 
import kmapper as km

# scp MapperClass_Proj-TSNE_Clstr-DBSCAN-1pt5_Date-202109120_NLim-330000 schaerf@iccluster040.iccluster.epfl.ch:/scratch/students/schaerf/mapper/replicasubset_finetune_keplermapper_output.html  C:OneDrive\Desktop
from mapper_tools import MapperGraph

N_LIMIT = 10000
DATE = '202109120'
PROJECTION = 'Proj-TSNE_Clstr-DBSCAN-1pt5'

path = '../data/'
data_dir = '../data/'
subfolder_dir = '01-06-2022'

#data = np.load(path + 'Replica_UIDs_ResNet_VGG_All.npy',allow_pickle=True)
data = np.load(data_dir + subfolder_dir + "/resnext-101_" + subfolder_dir +".npy", allow_pickle=True)
                
signatures_resnet = np.stack(data[:, 1])[:N_LIMIT]
#signatures_vgg = np.stack(data[:, 2])[:N_LIMIT]
print('\n> (1/6) Data loaded properly \n')

mapper_graph = MapperGraph(signatures_resnet, N_LIMIT, projection=sklearn.manifold.TSNE(n_components= 2),\
                           cover = km.Cover(n_cubes=3, perc_overlap=0.35), 
                           clustering = sklearn.cluster.OPTICS(max_eps=0.5, min_samples=1)
                           ) # 280, 0.1, 1.5 subset 30
print('\n> (2/6) Mapper graph complete \n')

with open(data_dir + f'MapperClass_{PROJECTION}_Date-{DATE}_NLim-{N_LIMIT}', 'wb') as f:
    pickle.dump(mapper_graph, f)
print('\n> (3/6) Saved mapper graph \n')

mapper_graph.loss_value_2()
print('\n> (4/6) Loss value computed \n')

with open(data_dir + f'MapperClass_{PROJECTION}_Date-{DATE}_NLim-{N_LIMIT}', 'wb') as f:
    pickle.dump(mapper_graph, f)
print('\n> (5/6) Saved mapper graph \n')

print(mapper_graph.morpho_graph.head())
# with open('mapper_class_20210908', 'rb') as f:
#     mapper_graph = pickle.load(f)

list_good_nodes = mapper_graph.morpho_graph.loc[(mapper_graph.morpho_graph['path_dist_mapper']==0)]

good_table = []
for index, node in list_good_nodes.iterrows():
    for cluster_id in node['node1']:
        cluster_members = mapper_graph.graph["nodes"][cluster_id]
        uid_sub = mapper_graph.df.loc[mapper_graph.df.uid.isin(mapper_graph.uids[cluster_members])][['uid']]
        x = [len(cluster_members),uid_sub]

        good_table.append([mapper_graph.graph['meta_data'], x])

np.save(data_dir + f'NullDistNodes_{PROJECTION}_Date-{DATE}_NLim-{N_LIMIT}.npy', good_table)
print('\n> (6/6) Computed null distance nodes \n')

mapper_graph.visualize_graph()
