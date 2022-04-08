import sys
from flask import Flask, render_template, request
from requests import get

sys.path.insert(0, ".")

from annotation_loop import get_links, setup, store_morph

embeddings, data, tree, reverse_map, uid2path = setup(
    data_dir="./data/", path="./data/", size=10000
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
    if request.method == "POST":
        if request.form["submit"] in ["text_search", "random_search"]:
            if request.form["submit"] == "text_search":
                image_uid = request.form["item"]
            else:
                image_uid = False

            compared_image, similar_images = get_links(
                embeddings, data, tree, reverse_map, uid2path, uid=image_uid, n=7
            )

            image_uid, compared_with_img_url = compared_image

            number_of_results = len(similar_images)

        if request.form["submit"] == "similar_images":
            similar_imges_uids = []
            for form_key in request.form.keys():
                if "ckb" in form_key:
                    similar_imges_uids.append(request.form[form_key])
            store_morph(request.form["UID_A"], similar_imges_uids, data_dir="./data/")

    return render_template(
        "annotate.html",
        results=similar_images,
        uploaded_image_url=compared_with_img_url,
        number_of_results=number_of_results,
        item=image_uid,
        cold_start=request.method == "GET",
    )


if __name__ == "__main__":

    app.run(port=8080)
    # from waitress import serve

    # serve(app, host="0.0.0.0", port=8080)
