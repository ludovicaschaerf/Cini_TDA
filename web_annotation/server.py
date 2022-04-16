from flask import Flask, render_template, request
import argparse

from annotation_tools import get_links, setup, store_morph

parser = argparse.ArgumentParser(description='Model specifics')
parser.add_argument('--n_subset', dest='n_subset',
                    type=int, help='', default=10000)
parser.add_argument('--n_show', dest='n_show',
                    type=int, help='', default=10)
parser.add_argument('--data_dir', dest='data_dir',
                    type=str, help='', default="./data/")
parser.add_argument('--path', dest='path',
                    type=str, help='', default="./data/")

args = parser.parse_args()

embeddings, data, tree, reverse_map, uid2path = setup(
    data_dir=args.data_dir, path=args.path, size=args.n_subset
)

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


if __name__ == "__main__":

    app.run(port=8080)

    # from waitress import serve

    # serve(app, host="0.0.0.0", port=8080)
