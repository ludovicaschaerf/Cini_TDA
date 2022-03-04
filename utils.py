import torchvision.transforms as transforms
from PIL import Image
from IPython.display import Image as Image2
import pickle
import pandas as pd
import networkx as nx
import numpy as np
from sklearn.neighbors import NearestNeighbors


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

    return positives


def preprocess_image(img_name):
    img = Image.open(img_name)
    tfms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return tfms(img).unsqueeze(0)


def get_embedding(img, model):
    embedding = model(img).detach().numpy().T
    norm = np.linalg.norm(embedding)
    return embedding / norm


def make_tree(metadata, embeds_folder, set_name="train"):
    metadata = metadata[metadata["set"] == set_name]
    metadata = metadata.groupby("uid").first().reset_index()
    A = []
    for it in metadata['uid']:
        with open(embeds_folder + it, 'rb') as infile:
            elt = pickle.load(infile)
            A.append(list(elt[:, 0]))

    kdt = NearestNeighbors(n_neighbors=2, metric="euclidean").fit(A)

    return kdt


def find_most_similar(row, metadata, kdt, embeds_folder):
    B = list(metadata["uid"])

    with open(embeds_folder + row["uid"].values[0], 'rb') as infile:
        img = np.array(pickle.load(infile)).reshape(1, -1)
    
    cv = kdt.kneighbors(img)[1][0][1]
    return B[cv]


def show_most_similar(row, metadata, kdt, embeds_folder, set_name="train"):

    metadata = metadata[metadata["set"] == set_name]
    metadata = metadata.groupby("uid").first().reset_index()
    cv = find_most_similar(row, metadata, kdt, embeds_folder)

    print("reference image", row["uid"].values[0])
    display(
        Image2('/scratch/students/schaerf/' + set_name + "/" + row["uid"].values[0] + ".jpg", width=400, height=400)
    )

    if row["uid"].values[0] == row["img1"].values[0]:
        print("actual most similar image", row["img2"].values[0])
        display(
            Image2(
                '/scratch/students/schaerf/' + set_name + "/" + row["img2"].values[0] + ".jpg", width=400, height=400
            )
        )
    else:
        print("actual most similar image", row["img1"].values[0])
        display(
            Image2(
                '/scratch/students/schaerf/' + set_name + "/" + row["img1"].values[0] + ".jpg", width=400, height=400
            )
        )

    print("most similar image according to model", cv)
    display(Image2('/scratch/students/schaerf/' + set_name + "/" + cv + ".jpg", width=400, height=400))
