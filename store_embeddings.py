import pandas as pd
from torch import nn
import torchvision.models as models
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

data_dir = '/scratch/students/schaerf/'
data = pd.read_csv(data_dir + 'full_data.csv').drop(columns=['Unnamed: 0', 'level_0'])

def create_model(model_name, pooling):
    if model_name == "resnet50":
        model = models.resnet50(pretrained=True)
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

def preprocess_image_orig(img_name, resolution=480):
    img = Image.open(img_name)
    tfms = transforms.Compose(
        [
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return tfms(img).unsqueeze(0)


def get_embedding_orig(img, model, device='cpu'):
    embedding = model(img.squeeze(1).to(device))[0].cpu().detach().numpy()
    norm = np.linalg.norm(embedding)
    return embedding / norm


for model in ['resnet50', 'efficientnet0', 'efficientnet7']:
    for pool in ['max', 'avg']:
        for resolution in [240, 480]:
            print(model, pool, resolution)
            newmodel = create_model(model, pool)
            embeddings = [[uid, get_embedding_orig(preprocess_image_orig(data_dir + 'subset/' + uid + '.jpg', resolution), newmodel).squeeze(1).squeeze(1)] for uid in tqdm(data['uid'].unique())]
            np.save(data_dir + model + '_' + pool + '_' + str(resolution) + '.npy', np.array(embeddings, dtype=np.ndarray))
