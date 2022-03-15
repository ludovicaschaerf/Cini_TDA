import torchvision.transforms as transforms
from PIL import Image
from IPython.display import Image as Image2
import pickle
import pandas as pd
import networkx as nx
import numpy as np
from sklearn.neighbors import NearestNeighbors, BallTree


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


def make_tree(metadata, embeds):
    metadata = metadata.groupby("uid").first().reset_index()
    if type(list(embeds.values())[0]) == np.ndarray:
        A = np.array([np.array(embeds[it]) for it in metadata["uid"]])
    else:
        A = np.array([np.array(embeds[it].toarray()[0]) for it in metadata["uid"]])
    # kdt = NearestNeighbors(n_neighbors=n, metric="euclidean").fit(A)
    kdt = BallTree(A, metric="euclidean")

    return kdt


def find_most_similar(row, metadata, kdt, embeds, n=1):
    B = list(metadata["uid"])
    if type(list(embeds.values())[0]) == np.ndarray:
        img = embeds[row["uid"].values[0]].reshape(1, -1)
    else:
        img = embeds[row["uid"].values[0]].toarray()[0].reshape(1, -1)
    # cv = kdt.kneighbors(img)[1][0][:n]
    cv = kdt.query(img, k=n)[1][0]
    return [B[c] for c in cv]


def find_most_similar_embed(row_emb, metadata, kdt, embeds, n=1):
    B = list(metadata["uid"])
    img = row_emb.reshape(1, -1)

    cv = kdt.query(img, k=n)[1][0]
    return [B[c] for c in cv]


def show_most_similar(row, metadata, kdt, embeds, n=1):

    metadata = (
        metadata.groupby("uid")
        .agg({"img1": lambda x: list(x), "img2": lambda x: list(x), "set": "first"})
        .reset_index()
    )

    cv = find_most_similar(row, metadata, kdt, embeds, n=21)
    index = metadata.loc[metadata["uid"] == row["uid"].values[0]].index[0]

    print("reference image", row["uid"].values[0])
    print(row["AuthorOriginal"].values[0], row["Description"].values[0])
    display(
        Image2(
            "/scratch/students/schaerf/"
            + row["set"].values[0]
            + "/"
            + row["uid"].values[0]
            + ".jpg",
            width=400,
            height=400,
        )
    )

    try:
        most_sim = set(
            list(metadata.loc[index, "img2"]) + list(metadata.loc[index, "img1"])
        )
        most_sim.remove(row["uid"].values[0])
    except:
        most_sim = set(
            list(metadata.loc[index, "img2"]) + list(metadata.loc[index, "img1"])
        )

    print("actual most similar image", list(most_sim)[0])
    display(
        Image2(
            "/scratch/students/schaerf/"
            + row["set"].values[0]
            + "/"
            + list(most_sim)[0]
            + ".jpg",
            width=400,
            height=400,
        )
    )

    for i in range(1, n):
        print(i + 1, "th most similar image according to model", cv[i])
        display(
            Image2(
                "/scratch/students/schaerf/subset/" + cv[i] + ".jpg",
                width=400,
                height=400,
            )
        )

    print('Recall @ 20: ',  len(list(set(cv[1:]).intersection(most_sim))) / min(len(cv[1:]), len(list(most_sim))))
