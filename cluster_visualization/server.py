from flask import Flask, render_template, request
import pandas as pd
import argparse

from utils_clusters import make_clusters, images_in_clusters

parser = argparse.ArgumentParser(description='Model specifics')
# parser.add_argument('--n_subset', dest='n_subset',
#                     type=int, help='', default=10000)
# parser.add_argument('--n_show', dest='n_show',
#                     type=int, help='', default=10)
parser.add_argument('--data_dir', dest='data_dir',
                    type=str, help='', default="../data/")
# parser.add_argument('--path', dest='path',
#                     type=str, help='', default="./data/")

args = parser.parse_args()

cluster_df = make_clusters(args.data_dir)
data = pd.read_csv(args.data_dir + 'dedup_data.csv').drop(columns=['Unnamed: 0', 'level_0'])

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/clusters", methods=["GET", "POST"])
def clusters():
    
    INFO = images_in_clusters(cluster_df, data)
    if request.method == "POST":
        # if request.form["submit"] in ["text_search", "random_search"]:
        #     if request.form["submit"] == "text_search":
        #         image_uid = request.form["item"]
        #     else:
        #         image_uid = False

        #     compared_image, similar_images = get_links(
        #         embeddings, data, tree, reverse_map, uid2path, uid=image_uid, n=args.n_show + 1
        #     )

        #     image_uid, compared_with_img_url, info = compared_image

        #     number_of_results = len(similar_images)
        print('hey')
        
    return render_template(
        "clusters.html",
        data=INFO,
        cold_start=request.method == "GET",
    )


if __name__ == "__main__":

    app.run(port=8080)

    # from waitress import serve

    # serve(app, host="0.0.0.0", port=8080)
