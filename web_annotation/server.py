from flask import Flask, render_template, request
import argparse
from glob import glob
import json


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

parser.add_argument('--eps', dest='eps',
                    type=float, help='', default=0.5)

args = parser.parse_args()

embeddings, data, tree, reverse_map, uid2path = setup(
    data_dir=args.data_dir, path=args.path, size=args.n_subset
)

cluster_df_rerank = make_clusters(args.data_dir)
data_rerank = pd.read_csv(args.data_dir + 'dedup_data.csv').drop(columns=['Unnamed: 0', 'level_0'])


data_file = 'data_wga_cini_45000.csv' 
embeds_file = 'resnext-101_epoch_410-05-2022_10%3A11%3A05.npy'
map_file = 'map2pos_10-05-2022.pkl'
cluster_file = 'clusters_'+str(args.eps)+'_10-05-2022_19'

data_file = 'dedup_data_sample_wga.csv' 
embeds_file = 'resnext-101_epoch_901-05-2022_19%3A45%3A03.npy' 
map_file = 'map2pos.pkl'
cluster_file = 'clusters_'+str(args.eps)+'_01-05-2022_19'


hierarchical_file = 'dedup_data_sample_wga_cluster.csv'

if args.precomputed:
    with open(args.data_dir + cluster_file + '.pkl', 'rb') as infile:
        cluster_df = pickle.load(infile)
else:
    cluster_df = make_clusters_embeddings(args.data_dir, dist=args.eps, data_file=data_file, embed_file=embeds_file)

#data = pd.read_csv(args.data_dir + data_file).drop(columns=['Unnamed: 0', 'level_0'])

data = pd.read_csv(args.data_dir + hierarchical_file).drop(columns=['Unnamed: 0', ])


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

@app.route("/clusters_hierarchical", methods=["GET", "POST"])
def clusters_hierarchical():
    cluster_df_hierarchical = data.copy()
    cluster_df_hierarchical['cluster'] = cluster_df_hierarchical['cluster_desc']

    INFO = images_in_clusters(cluster_df_hierarchical, data, map_file=map_file)
        
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

@app.route("/clusters_embeds", methods=["GET", "POST"])
def clusters_embeds():
    
    INFO = images_in_clusters(cluster_df, data, map_file=map_file)
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


@app.route("/visual_clusters", methods=["GET", "POST"])
def visual_clusters():
    
    INFO = images_in_clusters(cluster_df, data, map_file=map_file)
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
        "visual_clusters.html",
        data=convert_to_json(INFO),
        cold_start=request.method == "GET",
    )


if __name__ == "__main__":

    app.run(port=8080)