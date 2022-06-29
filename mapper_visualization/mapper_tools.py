import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Image
from IPython.core.display import HTML, display
from IPython.display import IFrame
#import plotly.express as px
import kmapper as km
from kmapper import jupyter
from tqdm import tqdm
import sklearn
import pickle
import networkx as nx

path = '../data/'
data_dir = '../data/'
subfolder_dir = '01-06-2022'


class MapperGraph:

    
    df = pd.read_csv(data_dir + 'data_sample.csv')#.drop(columns=['Unnamed: 0', 'level_0'])
    df = df[df['path'].str.contains('cini')]
    df['ImageNumber'] = df['path'].apply(lambda x: x.split('/')[-1].split('_')[1].split('.')[0])
    #print(df.shape)
    
    mapper = km.KeplerMapper(verbose=1)

    def __init__(self, data, n, projection, cover, clustering):
           
        self.data = data 
        self.uids = np.load(data_dir + subfolder_dir + "/resnext-101_" + subfolder_dir +".npy", allow_pickle=True)[:,0][0:n]
        
        with open(path + 'save_link_data_2018_08_02.pkl', 'rb') as f:
            morpho_complete  = pickle.load(f)
        self.morpho_graph = morpho_complete.loc[((morpho_complete.img1.isin(self.uids)) & (morpho_complete.img2.isin(self.uids))) ][['uid', 'img1', 'img2', 'type', 'annotated']]
        self.morpho_graph['path_dist_mapper'] = -1

        self.proj = projection
        self.cover = cover
        self.clust = clustering
        self.mapper = km.KeplerMapper(verbose=1)
        self.graph = self.mapper_graph()
        self.graph_nx = self.create_network()
        self.diam = self.diameter()
        
    def __str__(self):
        msg = 'Mapper graph created with:'+ len(self.graph['nodes']) + 'nodes and '+ len(self.graph['links']) + 'edges'
        return msg

    def mapper_graph(self): 
        #mapper = km.KeplerMapper(verbose=1)
        projected_data = self.mapper.fit_transform(self.data, projection= self.proj)
        graph = self.mapper.map(projected_data, self.data, clusterer=self.clust, cover=self.cover, remove_duplicate_nodes=True)
        return graph


    def get_mapper_node(self, uid):
            mapper_node     = []
            index_painting  = np.where(self.uids == uid)[0]
            #print(index_painting)
            for node in self.graph['nodes']:
                if index_painting in self.graph['nodes'][node]:
                    mapper_node.append(node)
            return mapper_node

    def create_network(self):
        g = nx.Graph()
        g.add_nodes_from(self.graph['nodes'].keys())
        for k, v in self.graph['links'].items():
            g.add_edges_from(([(k, t) for t in v]))
        return g
        

    def diameter(self):
        components      = nx.connected_components(self.graph_nx)
        largest_component = max(components, key=len)
        subgraph        = self.graph_nx.subgraph(largest_component)
        return nx.diameter(subgraph)

    

    def get_path_dist(self, node1, node2): 
        path_length     = 1
        #print(node1, node2)
        for n1 in node1:
            for n2 in node2:
                #print(n1, n2)
                if n2 in nx.node_connected_component(self.graph_nx, n1):
                    path_length = np.minimum(path_length, nx.shortest_path_length(self.graph_nx, n1, n2)/self.diam)

        return path_length


    def see_img_in_cluster(self, cluster_id):
        cluster_members = []
        if cluster_id in self.graph["nodes"]:
            cluster_members = self.graph["nodes"][cluster_id]
        df_sub = self.df.loc[self.df.uid.isin(self.uids[cluster_members])][['Description', 'Drawer', 'ImageNumber', 'uid']]

        list_urls = [(uid, drawer, image, f'https://dhlabsrv4.epfl.ch/iiif_replica/cini%2F{drawer}%2F{drawer}_{image}.jpg/full/300,/0/default.jpg') for drawer, image, uid in list(zip(df_sub.Drawer, df_sub.ImageNumber, df_sub.uid))]
        for uid, drawer, image, url in list_urls:
            print(uid, drawer, image)
            display(Image(url= url, width=150, height=200))


    def update_morphograph(self):
        self.morpho_graph['node1'] = self.morpho_graph['img1'].apply(lambda x:self.get_mapper_node(x))
        self.morpho_graph['node2'] = self.morpho_graph['img2'].apply(lambda x:self.get_mapper_node(x))
        dist = []
        for n1, n2 in tqdm(list(zip(self.morpho_graph.node1,self.morpho_graph.node2))):
            #dist.append(self.get_path_dist( self.graph, self.diameter, n1, n2))
            dist.append(self.get_path_dist( n1, n2))
        self.morpho_graph['path_dist_mapper'] = dist


    def loss_value(self):
        self.update_morphograph()
        loss = self.morpho_graph.path_dist_mapper.sum()/len(self.morpho_graph)
        return loss

    def update_morphograph_2(self):
        print('starting update morphograph')
        self.morpho_graph['node1'] = self.morpho_graph['img1'].apply(lambda x:self.get_mapper_node(x))
        self.morpho_graph['node2'] = self.morpho_graph['img2'].apply(lambda x:self.get_mapper_node(x))
        dist = []
        
        idx = 0
        for n1, n2 in tqdm(list(zip(self.morpho_graph.node1,self.morpho_graph.node2))):
            idx +=1
            print(f"\t {idx} / {len(self.morpho_graph.node1)}")
            if n1 == n2:
                dist.append(0.0)
            else: dist.append(1.0)
        self.morpho_graph['path_dist_mapper'] = dist

    def loss_value_2(self):
        self.update_morphograph_2()
        loss = self.morpho_graph.path_dist_mapper.sum()/len(self.morpho_graph)
        return loss

    
    
    def get_good_nodes(self):
        good_nodes = self.morpho_graph.loc[(self.morpho_graph['path_dist_mapper']==0)]
        return good_nodes 

    def visualize_graph(self): # todo : fix this 
        html = self.mapper.visualize(self.graph,path_html=data_dir + "keplermapper.html",
                 title="replica(n_samples=8959)", 
                 custom_tooltips = np.array([f"<img src= 'https://dhlabsrv4.epfl.ch/iiif_replica/cini%2F{drawer}%2F{drawer}_{image}.jpg/full/300,/0/default.jpg'>" for (drawer, image) in pd.merge(pd.Series(self.uids, name = 'uid').to_frame(), self.df, on = 'uid', how = 'left')[['Drawer', 'ImageNumber']].values]))
