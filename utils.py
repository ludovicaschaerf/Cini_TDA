import torchvision.transforms as transforms
from PIL import Image
from IPython.display import Image as Image2
import pickle
import pandas as pd
import networkx as nx
import numpy as np
from sklearn.neighbors import NearestNeighbors, BallTree
#import umap
from sklearn.decomposition import PCA, TruncatedSVD
from tqdm import tqdm
from glob import glob
import matplotlib.pyplot as plt

data_dir = '/scratch/students/schaerf/'

with open(data_dir + 'uid2path.pkl', 'rb') as outfile:
    uid2path = pickle.load(outfile)


#########################################################
##### Create embeddings
#########################################################

def make_tree_orig(embeds, reverse_map = False):
    if reverse_map:
        kdt = BallTree(np.vstack(embeds[:,1]), metric="euclidean")
        reverse_map = {k:embeds[k,0] for k in range(len(embeds))}
        return kdt, reverse_map
    else:
        kdt = BallTree(np.vstack(embeds[:,1]), metric="euclidean")
        return kdt

def find_most_similar_orig(uid, tree, embeds, uids, n=401):
    img = np.vstack(embeds[embeds[:,0] == uid][:,1]).reshape(1, -1)
    cv = tree.query(img, k=n)[1][0]
    return [uids[c] for c in cv if uids[c] != uid]

def find_most_similar_no_theo(uid, tree, embeds, uids, list_theo, n=401):
    img = np.vstack(embeds[embeds[:,0] == uid][:,1]).reshape(1, -1)
    cv = tree.query(img, k=n)[1][0]
    return [uids[c] for c in cv if uids[c] not in list_theo] #not in uids_match

def find_pos_matches(uids_sim, uids_match, how='all'):
    matched = list(filter(lambda i: uids_sim[i] in uids_match, range(len(uids_sim))))
    while len(matched) < len(uids_match):
        matched.append(400)
    if how == 'all':
        if len(matched) > 0:
            return matched
        else:
            return [400]
    elif how == 'first':
        if len(matched) > 0:
            return matched[0]
        else:
            return 400
    elif how == 'median':
        if len(matched) > 0:
            return np.median(np.array(matched))
        else:
            return 400

def make_rank(uids_sim, uids_match):
    return [1 if uid in uids_match else 0 for uid in uids_sim]

def catch(x):
    try:
        return uid2path[x]
    except:
        return np.nan

#########################################################
##### Create train and test set
#########################################################


def get_train_test_split(metadata, morphograph):

    positives = morphograph[morphograph["type"] == "POSITIVE"]
    positives.columns = ["uid_connection", "img1", "img2", "type", "annotated"]

    # creating connected components
    G = nx.from_pandas_edgelist(
        positives,
        source="img1",
        target="img2",
        create_using=nx.DiGraph(),
        edge_key="uid_connection",
    )
    components = [x for x in nx.weakly_connected_components(G)]

    # merging the two files
    positives = pd.concat(
        [
            metadata.merge(positives, left_on="uid", right_on="img1", how="inner"),
            metadata.merge(positives, left_on="uid", right_on="img2", how="inner"),
        ],
        axis=0,
    ).reset_index()

    # adding set specification to df
    mapper = {it: number for number, nodes in enumerate(components) for it in nodes}
    positives["cluster"] = positives["uid"].apply(lambda x: mapper[x])
    positives["set"] = [
        "test" if cl % 3 == 0 else "train" for cl in positives["cluster"]
    ]

    positives['set'] = ['val' if i % 3 == 0 and cl == 'test' else cl for i,cl in enumerate(positives['set'])]

    return positives

def remove_duplicates(metadata, morphograph):

    positives = morphograph[morphograph["type"] == "DUPLICATE"]
    positives.columns = ["uid_connection", "img1", "img2", "type", "annotated"]

    # creating connected components
    G = nx.from_pandas_edgelist(
        positives,
        source="img1",
        target="img2",
        create_using=nx.DiGraph(),
        edge_key="uid_connection",
    )
    components = [x for x in nx.weakly_connected_components(G)]

    # merging the two files
    positives = pd.concat(
        [
            metadata.merge(positives, left_on="uid", right_on="img1", how="outer"),
            metadata.merge(positives, left_on="uid", right_on="img2", how="outer"),
        ],
        axis=0,
    ).reset_index()

    # adding set specification to df
    mapper = {it:it for it in positives["uid"].unique()}

    for number, nodes in enumerate(components):
        for it in nodes:
            mapper[it] = number
              
    positives["cluster"] = positives["uid"].apply(lambda x: mapper[x])
    positives = positives.groupby('cluster').first().reset_index()
    return positives

#########################################################
##### Processing
#########################################################


def show_images(img_names):
    f, axarr = plt.subplots(1,3, figsize=(20,10))
    axarr = axarr.flatten()
    for i,name in enumerate(img_names):
        img = Image.open(name)
        axarr[i].imshow(img) 
        
    plt.show()
    
        
def preprocess_image(img_name, resolution=480):
    img = Image.open(img_name)
    tfms = transforms.Compose(
        [
            transforms.ToTensor(),
            #transforms.RandomResizedCrop((resolution, resolution), ),
            transforms.Resize((resolution, resolution)),
            #transforms.ColorJitter(
            #        brightness=0.4,
            #        contrast=0.4,
            #        saturation=0.4
            #),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            
        ]
    )
    return tfms(img).unsqueeze(0)


def get_embedding(img, model, type='fine_tune', device='cpu'):
    if type == 'fine_tune':
        embedding = model.predict(img.squeeze(1).to(device))[0].cpu().detach().numpy()
    else:
        embedding = model(img.squeeze(1).to(device))[0].cpu().detach().numpy()
        norm = np.linalg.norm(embedding)
        embedding = embedding / norm
    return embedding 

def get_lower_dimension(embeddings, dimensions=100, method='umap'):
    if method == 'umap':
        embeddings_new = umap.UMAP(n_components=dimensions, metric='cosine').fit(np.vstack(embeddings[:,1]))
        embeddings_new = embeddings_new.embedding_
    elif method == 'pca':
        embeddings_new = PCA(n_components=dimensions).fit_transform(np.vstack(embeddings[:,1]))
    elif method == 'svd':
        embeddings_new = TruncatedSVD(n_components=dimensions).fit_transform(np.vstack(embeddings[:,1]))
    
    return embeddings_new


#########################################################
##### Deprecated
#########################################################


def make_tree(metadata, embeds):
    metadata = metadata.groupby("uid").first().reset_index()
    if type(list(embeds.values())[0]) == np.ndarray:
        A = np.array([np.array(embeds[it]) for it in metadata['uid']])
    else:
        A = np.array([np.array(embeds[it].toarray()[0]) for it in metadata['uid']])
    #kdt = NearestNeighbors(n_neighbors=n, metric="euclidean").fit(A)
    kdt = BallTree(A, metric="euclidean")

    return kdt

def make_tree_list(embeds):
    kdt = BallTree(embeds, metric="euclidean")
    return kdt


def find_most_similar(row, metadata, kdt, embeds, n=1):
    B = list(metadata["uid"])
    if type(list(embeds.values())[0]) == np.ndarray:
        img = embeds[row["uid"].values[0]].reshape(1, -1)
    else:
        img = embeds[row["uid"].values[0]].toarray()[0].reshape(1, -1)
    #cv = kdt.kneighbors(img)[1][0][:n]
    cv = kdt.query(img, k=n)[1][0]
    return [B[c] for c in cv]

def find_most_similar_list(uid, kdt, embeds, uids, uids_match, n=20):
    img = embeds[np.where(np.array(uids) == uid)[0]].reshape(1, -1)
    cv = kdt.query(img, k=n)[1][0]
    return [uids[c] for c in cv if uids[c] not in uids_match]

def show_most_similar(row, metadata, kdt, embeds, n=1):

    metadata = metadata.groupby("uid").first().reset_index()
    cv = find_most_similar(row, metadata, kdt, embeds, n=n)

    print("reference image", row["uid"].values[0], row["AuthorOriginal"].values[0], row["Description"].values[0])
    display(
        Image2('/scratch/students/schaerf/' + row["set"].values[0] + "/" + row["uid"].values[0] + ".jpg", width=400, height=400)
    )

    if row["uid"].values[0] == row["img1"].values[0]:
        print("actual most similar image", row["img2"].values[0])
        display(
            Image2(
                '/scratch/students/schaerf/' + row["set"].values[0] + "/" + row["img2"].values[0] + ".jpg", width=400, height=400
            )
        )
    else:
        print("actual most similar image", row["img1"].values[0])
        display(
            Image2(
                '/scratch/students/schaerf/' + row["set"].values[0] + "/" + row["img1"].values[0] + ".jpg", width=400, height=400
            )
        )

    for i in range(len(cv)):
        print(i+1, "th most similar image according to model", cv[i])
        display(Image2('/scratch/students/schaerf/subset/' + cv[i] + ".jpg", width=400, height=400))


def make_training_set(data_dir, model, subset, device='cpu'):
    embeddings = [get_embedding(preprocess_image(data_dir + uid + '.jpg'), model, device=device).squeeze(3).squeeze(1).squeeze(0) for uid in tqdm(subset['uid'].unique())]
    embeddings_new = get_lower_dimension(embeddings)
    tree = make_tree_list(embeddings_new)

    train_test = subset[subset['set'].notnull()].drop(columns=['level_0']).reset_index()

    Cs = []
    for i in tqdm(range(train_test.shape[0])):
        list_theo = list(train_test[train_test['img1'] == train_test['uid'][i]]['img2']) + list(train_test[train_test['img2'] == train_test['uid'][i]]['img1']) + [train_test['uid'][i]]
        list_ = find_most_similar_list(train_test['uid'][i], tree, embeddings_new, list(subset['uid'].unique()), list_theo)
        Cs.append(list_)

    train_test['C'] = Cs

    list_downloaded = [file.split('/')[-1].split('.')[0] for file in glob(data_dir + 'subset/*')]


    train_test = train_test[train_test['img1'].isin(list_downloaded)]
    train_test = train_test[train_test['img2'].isin(list_downloaded)]

    final = train_test[['img1', 'img2', 'C', 'set']].explode('C')
    final.columns = ['A', 'B', 'C', 'set']
    
    final[final['set'] == 'train'].reset_index().to_csv(data_dir + 'abc_train.csv')
    final[final['set'] == 'test'].reset_index().to_csv(data_dir + 'abc_test.csv')

    return final
