from flask import Flask, render_template, request
import argparse
from glob import glob

from annotation_tools import get_links, setup, store_morph
from utils_clusters import * 

parser = argparse.ArgumentParser(description='Model specifics')
parser.add_argument('--n_subset', dest='n_subset',
                    type=int, help='', default=10000)
parser.add_argument('--n_show', dest='n_show',
                    type=int, help='', default=10)

parser.add_argument('--path', dest='path',
                    type=str, help='', default="../data/")

parser.add_argument('--data_dir', dest='data_dir',
                    type=str, help='', default="../data/")

parser.add_argument('--precomputed', dest='precomputed',
                    type=bool, help='', default=True)


args = parser.parse_args()

embeddings, data, tree, reverse_map, uid2path = setup(
    data_dir=args.data_dir, path=args.path, size=args.n_subset
)

cluster_df_rerank = make_clusters(args.data_dir)
data_rerank = pd.read_csv(args.data_dir + 'dedup_data.csv').drop(columns=['Unnamed: 0', 'level_0'])

if args.precomputed:
    with open(args.data_dir + 'clusters_0.5_01-05-2022_19.pkl', 'rb') as infile:
        cluster_df = pickle.load(infile)
else:
    cluster_df = make_clusters_embeddings(args.data_dir)

cluster_file = 'clusters_0.5_01-05-2022_19'
data = pd.read_csv(args.data_dir + 'dedup_data_sample_wga.csv')

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/annotate", methods=["GET", "POST"])
def annotate_images():
    similar_images = []
    number_of_results = 0
    image_uid = ""
    compared_with_img_url = ""
    info = ""
    if request.method == "POST":
        if request.form["submit"] in ["text_search", "random_search"]:
            if request.form["submit"] == "text_search":
                image_uid = request.form["item"]
            else:
                image_uid = False

            compared_image, similar_images = get_links(
                embeddings, data, tree, reverse_map, uid2path, uid=image_uid, n=args.n_show + 1
            )

            image_uid, compared_with_img_url, info = compared_image

            number_of_results = len(similar_images)

        if request.form["submit"] == "similar_images":
            similar_imges_uids = []
            for form_key in request.form.keys():
                if "ckb" in form_key:
                    similar_imges_uids.append(request.form[form_key])
            store_morph(request.form["UID_A"],
                        similar_imges_uids, data_dir=args.data_dir)

    return render_template(
        "annotate.html",
        results=similar_images,
        uploaded_image_url=compared_with_img_url,
        number_of_results=number_of_results,
        item=image_uid,
        info=info,
        cold_start=request.method == "GET",
    )

@app.route("/clusters_rerank", methods=["GET", "POST"])
def clusters():
    
    INFO = images_in_clusters(cluster_df_rerank, data_rerank)
        
    return render_template(
        "clusters.html",
        data=INFO,
        cold_start=request.method == "GET",
    )

@app.route("/clusters_embeds", methods=["GET", "POST"])
def clusters_embeds():
    
    INFO = images_in_clusters(cluster_df, data)
    if request.method == "POST":
        if request.form["submit"] == "similar_images":
                       
            
            imges_uids_sim = []
            for form_key in request.form.keys():
                if "ckb" in form_key:
                    imges_uids_sim.append(request.form[form_key])
            cluster_num = int(request.form["form"])
            
            store_morph_cluster(imges_uids_sim, INFO[int(request.form["form"])], cluster_num, cluster_file, data_dir=args.data_dir)

        if request.form["submit"] == "both_images":
        
            imges_uids_sim = []
            for form_key in request.form.keys():
                if "ckb" in form_key:
                    imges_uids_sim.append(request.form[form_key])
            cluster_num = int(request.form["form"])
            
            store_morph_cluster_negatives(imges_uids_sim, INFO[int(request.form["form"])], cluster_num, cluster_file, data_dir=args.data_dir)


    return render_template(
        "clusters.html",
        data=INFO,
        cold_start=request.method == "GET",
    )


if __name__ == "__main__":

    app.run(port=8080)