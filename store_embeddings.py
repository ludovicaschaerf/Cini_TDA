import pandas as pd
from torch import nn
import torchvision.models as models
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from metrics import * #recall_at_k, mean_average_precision
from utils import * 
from IPython.display import Image as Image2
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def create_model(model_name, pooling):
    if model_name == "resnet50":
        model = models.resnet50(pretrained=True)
    elif model_name == "resnet101":
        model = models.resnet101(pretrained=True)
    elif model_name == "resnet152":
        model = models.resnet152(pretrained=True)
    elif model_name == 'densenet161':
        model = models.densenet161(pretrained=True)
    elif model_name == 'resnext-101':
        model = models.resnext101_32x8d(pretrained=True)
    elif model_name == 'regnet_x_32gf':
        model = models.regnet_y_32gf(pretrained=True)
    elif model_name == 'vit_b_16':
        model = models.vit_b_16(pretrained=True)
    elif model_name == 'convnext_tiny':
        model = models.convnext_tiny(pretrained=True)
    elif model_name == "efficientnet0":
        model = models.efficientnet_b0(pretrained=True)
    elif model_name == "efficientnet7":
        model = models.efficientnet_b7(pretrained=True)

    if pooling == "avg":
        newmodel = torch.nn.Sequential(
            *(list(model.children())[:-2]), nn.AdaptiveAvgPool2d((1, 1))
        )
    elif pooling == 'max':
        newmodel = torch.nn.Sequential(
            *(list(model.children())[:-2]), nn.AdaptiveMaxPool2d((1, 1), )
        )
    
    return newmodel



def get_scores(embeddings, train_test, data, list_downloaded=False, reverse_map=False):
    if reverse_map:
        tree, reverse_map = make_tree_orig(embeddings, True)
    
    else:
        tree = make_tree_orig(embeddings)
        reverse_map = list(data['uid'].unique())
    Cs = []
    Bs = []
    pos = []
    ranks = []

    if not list_downloaded:
        list_downloaded = list(train_test["img1"]) + list(train_test["img2"])

    for i in tqdm(range(train_test.shape[0])):
        if (train_test["img1"][i] in list_downloaded) and (train_test["img2"][i] in list_downloaded) & (train_test["set"][i] == 'test'):
            list_theo = (
                list(train_test[train_test["img1"] == train_test["uid"][i]]["img2"])
                + list(train_test[train_test["img2"] == train_test["uid"][i]]["img1"])
                + [train_test["uid"][i]]
            )
            Bs.append(list_theo)
            
            list_sim = find_most_similar_orig(
                train_test["uid"][i], tree, embeddings, reverse_map, n=min(data.shape[0], 4000)
            )
            Cs.append(list_sim[:400])
            matches = find_pos_matches(list_sim[:400], list_theo, how="all")
            pos.append(matches)
            rank = make_rank(list_sim, list_theo)
            ranks.append(rank)
            
    posses = [po for p in pos for po in p]
    posses_min = [p[0] for p in pos]
    posses_med = [np.median(np.array(p)) for p in pos]

    mean_position = np.mean(np.array(posses))
    mean_min_position = np.mean(np.array(posses_min))
    mean_median_position = np.mean(np.array(posses_med))
            
    print('all positions', mean_position)
    print('min positions', mean_min_position)
    print('median positions', mean_median_position)

    map = mean_average_precision(ranks)
    print('mean average precision', map)

    recall_400 = np.mean([recall_at_k(ranks[i], 400) for i in range(len(ranks))])
    recall_200 = np.mean([recall_at_k(ranks[i], 200) for i in range(len(ranks))])
    recall_100 = np.mean([recall_at_k(ranks[i], 100) for i in range(len(ranks))])
    recall_50 = np.mean([recall_at_k(ranks[i], 50) for i in range(len(ranks))])
    recall_20 = np.mean([recall_at_k(ranks[i], 20) for i in range(len(ranks))])
    print('recall @ 400', recall_400)
    print('recall @ 200', recall_200)
    print('recall @ 100', recall_100)
    print('recall @ 50', recall_50)
    print('recall @ 20', recall_20)

    return mean_position, mean_min_position, mean_median_position, map, recall_400, recall_200, recall_100, recall_50, recall_20

def make_training_set_orig(embeddings, train_test, data, data_dir, epoch=False, n=10):
    tree = make_tree_orig(embeddings)
    Cs = []
    for i in tqdm(range(train_test.shape[0])):
            list_theo = (
                list(train_test[train_test["img1"] == train_test["uid"][i]]["img2"])
                + list(train_test[train_test["img2"] == train_test["uid"][i]]["img1"])
                + [train_test["uid"][i]]
            )
            list_sim = find_most_similar_no_theo(
                train_test["uid"][i], tree, embeddings, list(data["uid"].unique()), list_theo, n=n+1
            )
            Cs.append(list_sim)
            

    
    train_test['C'] = Cs

    #list_downloaded = [file.split('/')[-1].split('.')[0] for file in glob(data_dir + 'subset/*')]

    #train_test = train_test[train_test['img1'].isin(list_downloaded)]
    #train_test = train_test[train_test['img2'].isin(list_downloaded)]

    final = train_test[['img1', 'img2', 'C', 'set']].explode('C')
    final.columns = ['A', 'B', 'C', 'set']
    final['A_path'] = final['A'].apply(lambda x: catch(x))
    final['B_path'] = final['B'].apply(lambda x: catch(x))
    final['C_path'] = final['C'].apply(lambda x: catch(x))
    
    final = final[final['C_path'].notnull() & final['A_path'].notnull() & final['B_path'].notnull()]
    print(final.shape)
    print(final.tail())

    if epoch:
        final[final['set'] == 'train'].reset_index().to_csv(data_dir + 'dataset/abc_train_' + str(epoch) + '.csv')
        final[final['set'] == 'test'].reset_index().to_csv(data_dir + 'dataset/abc_test_' + str(epoch) + '.csv')
        final[final['set'] == 'val'].reset_index().to_csv(data_dir + 'dataset/abc_val_' + str(epoch) + '.csv')
    else:
        final[final['set'] == 'train'].reset_index().to_csv(data_dir + 'dataset/abc_train.csv')
        final[final['set'] == 'test'].reset_index().to_csv(data_dir + 'dataset/abc_test.csv')
        final[final['set'] == 'val'].reset_index().to_csv(data_dir + 'dataset/abc_val.csv')

    return final


def show_similars(row, embeddings, train_test, data):
    
    tree = make_tree_orig(embeddings)
    
    list_theo = (
        list(train_test[train_test["img1"] == row["uid"].values[0]]["img2"])
        + list(train_test[train_test["img2"] == row["uid"].values[0]]["img1"])
        #+ [row["uid"].values[0]]
    )

    theo = list(set(list_theo))[0]
            
    sim = find_most_similar_orig(
        row["uid"].values[0], tree, embeddings, list(data["uid"].unique()), n=4
    )

    print("reference image", row["uid"].values[0], row["AuthorOriginal"].values[0], row["Description"].values[0])
    display(
        Image2('/scratch/students/schaerf/subset/' + row["uid"].values[0] + ".jpg", width=400, height=400)
    )

    print("actual most similar image", theo)
    display(
        Image2(
            '/scratch/students/schaerf/subset/' + theo + ".jpg", width=400, height=400
        )
    )
    
    for i in range(len(sim)):
        print(i+1, "th most similar image according to model", sim[i])
        display(Image2('/scratch/students/schaerf/subset/' + sim[i] + ".jpg", width=400, height=400))
   

def show_suggestions(row, embeddings, train_test):
    replica_dir = '/mnt/project_replica/datasets/cini/'

    tree, reverse_map = make_tree_orig(embeddings, reverse_map=True)
    
    if row["set"].values[0] in ['train', 'test']:
        list_theo = (
            list(train_test[train_test["img1"] == row["uid"].values[0]]["img2"])
            + list(train_test[train_test["img2"] == row["uid"].values[0]]["img1"])
            + [row["uid"].values[0]]
        )
    else:
        list_theo = [row["uid"].values[0]]
        
    sim = find_most_similar_no_theo(
        row["uid"].values[0], tree, embeddings, reverse_map, list_theo, n=8
    )


    f, axarr = plt.subplots(2,4, figsize=(30,10))
    img_A = mpimg.imread(replica_dir + row["path"].values[0])
    axarr = axarr.flatten()
    axarr[0].imshow(img_A)
    axarr[0].set_title(row["AuthorOriginal"].values[0] + row["Description"].values[0])
    for i in range(len(sim)):
        axarr[i+1].set_title(str(i) + "th most similar image" + sim[i])
        axarr[i+1].imshow(mpimg.imread(replica_dir + catch(sim[i])))
    
    plt.show()
    return row["uid"].values[0], sim
    

def main(models, pools, resolutions):
    data_dir = '/scratch/students/schaerf/'
    data = pd.read_csv(data_dir + 'full_data.csv').drop(columns=['Unnamed: 0', 'level_0'])
    for model in models: # ['resnext-101']'resnet50', 'efficientnet0', 'efficientnet7', 'resnet101', 'resnet152', 'densenet161', 'resnext-101', 'regnet_x_32gf', 
        for pool in pools: #'max', ['avg'] 
            for resolution in resolutions: #240, 480 [620]
                print(model, pool, resolution)
                newmodel = create_model(model, pool)
                embeddings = [[uid, get_embedding(preprocess_image(data_dir + 'subset/' + uid + '.jpg', resolution), newmodel).squeeze(1).squeeze(1)] for uid in tqdm(data['uid'].unique())]
                np.save(data_dir + 'models/' + model + '_' + pool + '_' + str(resolution) + '.npy', np.array(embeddings, dtype=np.ndarray))
