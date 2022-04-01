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

path = '/home/guhennec/scratch/2021_Cini/TopologicalAnalysis_Cini/data/'
data_dir = '/scratch/students/schaerf/'


class MapperGraph:

    
    df = pd.read_csv(path + 'Cini_20210811.csv', sep=';', low_memory=False) 
    df = pd.read_csv(data_dir + 'full_data.csv').drop(columns=['Unnamed: 0', 'level_0'])
    #print(df.shape)
    
    mapper = km.KeplerMapper(verbose=1)

    def __init__(self, data, n, projection, cover, clustering):
           
        self.data = data 
        #self.uids = np.load(path + 'Replica_UIDs_ResNet_VGG_All.npy',allow_pickle=True)[:,0][0:n]
        #self.uids = np.load(data_dir + 'embeddings/resnext-101_avg_480.npy', allow_pickle=True)[:,0][0:n]
        self.uids = np.load(data_dir + 'embeddings/resnext-101_epoch_0_31_03.npy', allow_pickle=True)[:,0][0:n]
        
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
        html = self.mapper.visualize(self.graph,path_html=data_dir + "mapper/replicasubset_finetune_keplermapper_output.html",
                 title="replica10000(n_samples=10000)", 
                 custom_tooltips = np.array([f"<img src= 'https://dhlabsrv4.epfl.ch/iiif_replica/cini%2F{drawer}%2F{drawer}_{image}.jpg/full/300,/0/default.jpg'>" for (drawer, image) in pd.merge(pd.Series(self.uids, name = 'uid').to_frame(), self.df, on = 'uid', how = 'left')[['Drawer', 'ImageNumber']].values]))


    
#graph = MapperGraph(data = data, projection=sklearn.decomposition.PCA(n_components= 2), cover =  km.Cover(n_cubes=20, perc_overlap=0.15), clustering = sklearn.cluster.KMeans(20))
#print(graph)


class ClusterGraph():

    #uids = np.load(path + 'Replica_UIDs_ResNet_VGG_All.npy',allow_pickle=True)[:,0] 
    uids = np.load(data_dir + 'embeddings/resnext-101_avg_480.npy', allow_pickle=True)[:,0]
            
    # df = pd.read_csv(path + 'Cini_20210811.csv', sep=';', low_memory=False)
    df = pd.read_csv(data_dir + 'full_data.csv').drop(columns=['Unnamed: 0', 'level_0'])
    
    with open(path + 'save_link_data_2018_08_02.pkl', 'rb') as f:
        morpho_complete  = pickle.load(f)
    morpho_graph = morpho_complete.loc[((morpho_complete.img1.isin(uids)) & (morpho_complete.img2.isin(uids))) ][['uid', 'img1', 'img2', 'type', 'annotated']]

    def __init__(self, data, clustering, threshold, parameter = None):
        self.data = data 
        self.clust = clustering
        self.param = parameter
        self.graph = self.clust_graph(threshold)
        self.diam = self.diameter()
        
    def __str__(self):
        msg = 'Clustering graph created with:'+ str(len(self.graph.nodes())) + 'nodes and '+ str(len(self.graph.edges())) + 'edges'
        return msg

    
    def clust_graph(self, threshold): 
        model = self.clust(self.param)
        model.fit(self.data)
        yhat = model.predict(self.data)
        clusters = np.unique(yhat)
        means = []

        G = nx.Graph()
        #G.add_nodes_from(clusters)
        #nx.set_node_attributes(G, name = "nodes_list")

        for cluster in clusters:
            row_ix = np.where(yhat == cluster)
            means.append(np.mean(self.data[row_ix], axis = 0))
            print(cluster)
            print(row_ix)
            G.add_node(node_for_adding=cluster, nodes_list=row_ix)
        
        for i in range(len(means)): 
            for j in range(len(means)):
                if (np.linalg.norm(means[i]-means[j])< threshold):
                    G.add_edge(i,j)

        return G

    def diameter(self):
        components      = nx.connected_components(self.graph)
        largest_component = max(components, key=len)
        subgraph        = self.graph.subgraph(largest_component)
        return nx.diameter(subgraph)



    def get_cluster_node_from_uid(self, uid):
        cluster_node     = []
        index_painting  = np.where(self.uids == uid)[0][0]
        #print(index_painting)
        #print("test", self.graph.nodes()) 
        #print("test nodes list", self.graph.nodes()[0]['nodes_list']) 
        
        for node in self.graph.nodes():
            #print(node)
            if index_painting in self.graph.nodes[node]["nodes_list"][0]:
                #print('yes')
                cluster_node.append(node)

        return cluster_node



    def get_path_dist(self, node1, node2): 
        path_length     = 1
        #print(node1, node2)
        for n1 in node1:
            for n2 in node2:
                #print(n1, n2)
                if n2 in nx.node_connected_component(self.graph, n1):
                    path_length = np.minimum(path_length, nx.shortest_path_length(self.graph, n1, n2)/self.diam)

        return path_length


    def see_img_in_cluster(self, cluster_id):

        cluster_members = []
        
        if cluster_id in self.graph.nodes():
            cluster_members = self.graph.nodes[cluster_id]["nodes_list"]

        df_sub = self.df.loc[self.df.uid.isin(self.uids[cluster_members])][['Description', 'Drawer', 'ImageNumber', 'uid']]

        list_urls = [(uid, drawer, image, f'https://dhlabsrv4.epfl.ch/iiif_replica/cini%2F{drawer}%2F{drawer}_{image}.jpg/full/300,/0/default.jpg') for drawer, image, uid in list(zip(df_sub.Drawer, df_sub.ImageNumber, df_sub.uid))]
        for uid, drawer, image, url in list_urls:
            print(uid, drawer, image)
            display(Image(url= url, width=150, height=200))


    def update_morphograph(self):
        self.morpho_graph['node1'] = self.morpho_graph['img1'].apply(lambda x:self.get_cluster_node_from_uid(x))
        self.morpho_graph['node2'] = self.morpho_graph['img2'].apply(lambda x:self.get_cluster_node_from_uid(x))
        dist = []
        for n1, n2 in tqdm(list(zip(self.morpho_graph.node1,self.morpho_graph.node2))):
            dist.append(self.get_path_dist(n1, n2)) #TODO: verifier ca 
        self.morpho_graph['path_dist_mapper'] = dist


    def loss_value(self):
        self.update_morphograph()
        loss = self.morpho_graph.path_dist_mapper.sum()/len(self.morpho_graph)
        return loss



 
