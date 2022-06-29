from flask import Flask, render_template, request
import argparse


#from annotation_tools import get_links, setup, store_morph
from utils_clusters import * 
from interest_metrics import add_interest_scores
from metrics_clusters import update_morph

# arguments to select which version to visualise
parser = argparse.ArgumentParser(description='Model specifics')
parser.add_argument('--data_dir', dest='data_dir',
                    type=str, help='Directory containing all the subfolders of the different efforts', default="../data/")

parser.add_argument('--subfolder', dest='subfolder',
                    type=str, help='Subfolder containing the desired embeddings, clustering and mapping file', default="25-05-2022/")

parser.add_argument('--precomputed', dest='precomputed',
                    type=bool, help='Whether to use already produced clustering file or produce it on the fly', default=True)

parser.add_argument('--type', dest='type',
                    type=str, help='Clustering method, choice of kmeans, optics, dbscan, mix (kmeans with outlier removal)', default='kmeans')

parser.add_argument('--eps', dest='eps',
                    type=float, help='Main clustering parameter. eps for dbscan, max_eps for optics and num_clusters for kmeans', default=1500)

parser.add_argument('--scores', dest='scores',
                    type=str, help='Yes or no to include the sorting methods for the morphograph', default='yes')


args = parser.parse_args()

# reranking clusters
cluster_df_rerank = make_clusters_rerank(args.data_dir+'rerank/')
data_rerank = pd.read_csv(args.data_dir + 'original/dedup_data.csv').drop(columns=['Unnamed: 0', 'level_0'])

# morphograph
if args.scores == 'yes':
    morpho = add_interest_scores(args.data_dir, translate=False, new=True, precomputed=False)
else:
    print('using existing morphograph')
    morpho = update_morph(args.data_dir, '', new=True)
    
morpho_ = morpho.fillna('')
print(morpho_.shape)
print('num of unique images', len(list(set(list(morpho_['img1']) + list(morpho_['img2']))))) 

print('uids that have img2', morpho_[morpho_['uid'].isin(morpho_[morpho_['uid'].isin(list(morpho_['img1']))]['img2'])].shape)
new_morph = morpho_.groupby('uid').first().reset_index()
print('num of retained images', new_morph.shape[0]) 
print('uids that have img2 unique', new_morph[new_morph['uid'].isin(new_morph[new_morph['uid'].isin(list(new_morph['img1']))]['img2'])].shape)
# eps becomes number of clusters
if args.type in ['mix','kmeans','spectral_clustering']:
    args.eps = int(args.eps)

    
# clustering files
data_file = 'data_sample.csv' 
if args.subfolder == '07-06-2022/':
    data_file = 'data.csv' 
    
# loading the files indicated
data_norm = pd.read_csv(args.data_dir + data_file).drop(columns=['annotated', 'set']).merge(new_morph[['uid', 'annotated', 'set']], left_on='uid', right_on='uid', how='left')
embeds_file = args.subfolder + 'resnext-101_'+args.subfolder.strip('/') +'.npy' 
map_file = args.subfolder + 'map2pos.pkl'
cluster_file = args.subfolder + 'clusters_'+args.type+'_'+str(args.eps)+'_'+args.subfolder.strip('/')+'_19'
    

with open(args.data_dir + cluster_file + '.pkl', 'rb') as infile:
    cluster_df = pickle.load(infile)
    cluster_df = cluster_df.sort_values('cluster')


app = Flask(__name__)

# simple home page
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/clusters_embeds", methods=["GET", "POST"])
def clusters_embeds():

    INFO, cluster = show_results_button(cluster_df, data_norm, map_file) 
    annotate_store(cluster_df, data_norm, map_file, cluster_file, args.data_dir) 
    
    return render_template(
        "clusters.html",
        item=cluster,
        data=INFO,
        cold_start=request.method == "GET",
    )


@app.route("/morphograph", methods=["GET", "POST"])
def morpho_show():
    
    print('morph clusters')
    print(new_morph['cluster'].nunique())
    clu2size = {i: cl for i,cl in zip(new_morph.groupby('cluster').size().index, new_morph.groupby('cluster').size().values)}
    new_morph['cluster_size'] = new_morph['cluster'].apply(lambda x: clu2size[x])
    new_morph_ = new_morph[new_morph['cluster_size'] > 1]
    print(new_morph_['cluster'].nunique())

    INFO = images_in_clusters(new_morph_, morpho_, map_file=map_file)
       
    if args.scores == 'yes':
        score_morph = {cluster: {col:group[col].values[0] for col in new_morph if 'scores' in col} for cluster, group in new_morph.groupby('cluster')}
        return render_template(
            "clusters.html",
            data=INFO,
            scores=score_morph,
            cold_start=request.method == "GET",
        )
    else:
        return render_template(
            "clusters.html",
            data=INFO,
            cold_start=request.method == "GET",
        )

@app.route("/clusters_rerank", methods=["GET", "POST"])
def clusters():
    
    INFO = images_in_clusters(cluster_df_rerank, data_rerank, map_file=map_file)
    annotate_store(cluster_df, data_norm, map_file, cluster_file, args.data_dir)

    return render_template(
        "clusters.html",
        data=INFO,
        cold_start=request.method == "GET",
    )



## deprecated 

# @app.route("/visual_clusters", methods=["GET", "POST"])
# def visual_clusters():
    
#     INFO = images_in_clusters(cluster_df, data_norm, map_file=map_file)
#     annotate_store(cluster_df, data_norm, map_file, cluster_file, args.data_dir)

#     return render_template(
#         "visual_clusters.html",
#         data=convert_to_json(INFO),
#         cold_start=request.method == "GET",
#     )


# parser.add_argument('--n_subset', dest='n_subset',
#                     type=int, help='', default=10)
# parser.add_argument('--n_show', dest='n_show',
#                     type=int, help='', default=10)


# # image retrieval
# embeddings, data, tree, reverse_map, uid2path = setup(
#     data_dir=args.data_dir, size=args.n_subset
# )


# @app.route("/annotate", methods=["GET", "POST"])
# def annotate_images():
#     similar_images = []
#     number_of_results = 0
#     image_uid = ""
#     compared_with_img_url = ""
#     info = ""
#     if request.method == "POST":
#         if request.form["submit"] in ["text_search", "random_search"]:
#             if request.form["submit"] == "text_search":
#                 image_uid = request.form["item"]
#             else:
#                 image_uid = False

#             compared_image, similar_images = get_links(
#                 embeddings, data, tree, reverse_map, uid2path, uid=image_uid, n=args.n_show + 1
#             )

#             image_uid, compared_with_img_url, info = compared_image

#             number_of_results = len(similar_images)

#         if request.form["submit"] == "similar_images":
#             similar_imges_uids = []
#             for form_key in request.form.keys():
#                 if "ckb" in form_key:
#                     similar_imges_uids.append(request.form[form_key])
#             store_morph(request.form["UID_A"],
#                         similar_imges_uids, data_dir=args.data_dir)

#     return render_template(
#         "annotate.html",
#         results=similar_images,
#         uploaded_image_url=compared_with_img_url,
#         number_of_results=number_of_results,
#         item=image_uid,
#         info=info,
#         cold_start=request.method == "GET",
#     )

# @app.route("/clusters_hierarchical", methods=["GET", "POST"])
# def clusters_hierarchical():
    
#     cluster_df_hierarchical = data.copy()
#     cluster_df_hierarchical['cluster'] = cluster_df_hierarchical['cluster_desc']

#     INFO = images_in_clusters(cluster_df_hierarchical, data, map_file=map_file)
#     annotate_store(cluster_df, data, map_file, cluster_file, args.data_dir) 
    
#     return render_template(
#         "clusters.html",
#         data=INFO,
#         cold_start=request.method == "GET",
#     )

# hierarchical files
# hierarchical_file = '01-05-2022/' + 'dedup_data_sample_wga_cluster.csv'
# data = pd.read_csv(args.data_dir + hierarchical_file).drop(columns=['Unnamed: 0', ])


if __name__ == "__main__":

    app.run(port=8080, )#debug=True)