from flask import Flask, render_template, request
import argparse
from glob import glob
import json


from annotation_tools import get_links, setup, store_morph
from utils_clusters import * 

parser = argparse.ArgumentParser(description='Model specifics')
parser.add_argument('--n_subset', dest='n_subset',
                    type=int, help='', default=1000)
parser.add_argument('--n_show', dest='n_show',
                    type=int, help='', default=10)

parser.add_argument('--data_dir', dest='data_dir',
                    type=str, help='', default="../data/")

parser.add_argument('--subfolder', dest='subfolder',
                    type=str, help='', default="01-05-2022/")

parser.add_argument('--precomputed', dest='precomputed',
                    type=bool, help='', default=True)

parser.add_argument('--type', dest='type',
                    type=str, help='', default='kmeans')

parser.add_argument('--eps', dest='eps',
                    type=float, help='', default=1500)

args = parser.parse_args()

# image retrieval
embeddings, data, tree, reverse_map, uid2path = setup(
    data_dir=args.data_dir, path=args.data_dir, size=args.n_subset
)

# reranking
cluster_df_rerank = make_clusters(args.data_dir+'rerank/')
data_rerank = pd.read_csv(args.data_dir + 'original/dedup_data.csv').drop(columns=['Unnamed: 0', 'level_0'])

# morphograph
morpho = pd.read_csv(args.data_dir + 'morphograph/morpho_dataset.csv')

if args.subfolder == '01-05-2022/':
    data_file = 'original/dedup_data_sample_wga.csv' 
    embeds_file = args.subfolder + 'resnext-101_epoch_901-05-2022_19%3A45%3A03.npy' 
    map_file = args.subfolder + 'map2pos.pkl'
    
    if args.type == 'dbscan':
        cluster_file = args.subfolder + 'clusters_'+str(args.eps)+'_01-05-2022_19'
    else:
        cluster_file = args.subfolder + 'clusters_'+args.type+'_'+str(int(args.eps))+'_01-05-2022_19'
    
    hierarchical_file = args.subfolder + 'dedup_data_sample_wga_cluster.csv'

    data = pd.read_csv(args.data_dir + hierarchical_file).drop(columns=['Unnamed: 0', ])

    
elif args.subfolder == '10-05-2022/':

    data_file = args.subfolder + 'data_wga_cini_45000.csv' 
    embeds_file = args.subfolder + 'resnext-101_epoch_410-05-2022_10%3A11%3A05.npy'
    map_file = args.subfolder + 'map2pos_10-05-2022.pkl'
    cluster_file = args.subfolder + 'clusters_'+str(args.eps)+'_10-05-2022_19'

    data_rerank = pd.read_csv(args.data_dir + 'original/dedup_data_sample_wga.csv').drop(columns=['Unnamed: 0', 'level_0'])



if args.precomputed:
    with open(args.data_dir + cluster_file + '.pkl', 'rb') as infile:
        cluster_df = pickle.load(infile)
        cluster_df = cluster_df.sort_values('cluster')
    
else:
    cluster_df = make_clusters_embeddings(args.data_dir, dist=args.eps, data_file=data_file, embed_file=embeds_file)




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


@app.route("/morphograph", methods=["GET", "POST"])
def morpho_show():
    

    INFO = images_in_clusters(morpho, morpho, map_file=map_file)
        

    return render_template(
        "clusters.html",
        data=INFO,
        cold_start=request.method == "GET",
    )


@app.route("/clusters_rerank", methods=["GET", "POST"])
def clusters():
    
    INFO = images_in_clusters(cluster_df_rerank, data_rerank, map_file=map_file)
        
    annotate_store(INFO, cluster_file, args.data_dir)

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

    
    annotate_store(INFO, cluster_file, args.data_dir) 
    
    return render_template(
        "clusters.html",
        data=INFO,
        cold_start=request.method == "GET",
    )

@app.route("/clusters_embeds", methods=["GET", "POST"])
def clusters_embeds():

    INFO, cluster = annotate_store_page(cluster_df, data, map_file) 
   
    annotate_store_special(cluster_df, data, map_file, cluster_file, args.data_dir) 
    
    return render_template(
        "clusters.html",
        item=cluster,
        data=INFO,
        cold_start=request.method == "GET",
    )


@app.route("/visual_clusters", methods=["GET", "POST"])
def visual_clusters():
    
    INFO = images_in_clusters(cluster_df, data, map_file=map_file)
    
    annotate_store(INFO, cluster_file, args.data_dir)

    return render_template(
        "visual_clusters.html",
        data=convert_to_json(INFO),
        cold_start=request.method == "GET",
    )


if __name__ == "__main__":

    app.run(port=8080, debug=True)