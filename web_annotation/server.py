import time

# from annotation_loop import main, get_data
from flask import Flask, redirect, url_for, render_template, request, jsonify, flash


import os
import json

from requests import get

# embeddings, data, tree, reverse_map = get_data(size=10000)

# print("len: {}".format(len(embeddings)))

app = Flask(__name__)

# current_directory = os.path.dirname(os.path.realpath(__file__))
# static_dir = os.path.join(current_directory, "static")

PAGE_SIZE = 6


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
            # photos, searched_items, number_of_results = search_photos(
            #     image_uid, page_number, PAGE_SIZE
            # )

            similar_images = [
                [
                    "id1",
                    f"https://dhlabsrv4.epfl.ch/iiif_replica/cini%2F61A%2F61A_632.jpg/full/300,/0/default.jpg",
                ],
                [
                    "id2",
                    f"https://dhlabsrv4.epfl.ch/iiif_replica/cini%2F87A%2F87A_507.jpg/full/300,/0/default.jpg",
                ],
                [
                    "id3",
                    f"https://dhlabsrv4.epfl.ch/iiif_replica/cini%2F1C%2F1C_484.jpg/full/300,/0/default.jpg",
                ],
                [
                    "id4",
                    f"https://dhlabsrv4.epfl.ch/iiif_replica/cini%2F42C%2F42C_694.jpg/full/300,/0/default.jpg",
                ],
                [
                    "id5",
                    f"https://dhlabsrv4.epfl.ch/iiif_replica/cini%2F96A%2F96A_336.jpg/full/300,/0/default.jpg",
                ],
                [
                    "id6",
                    f"https://dhlabsrv4.epfl.ch/iiif_replica/cini%2F7A%2F7A_161.jpg/full/300,/0/default.jpg",
                ],
            ]
            compared_with_img_url = f"https://dhlabsrv4.epfl.ch/iiif_replica/cini%2F97A%2F97A_64.jpg/full/300,/0/default.jpg"

            number_of_results = len(similar_images)

        if request.form["submit"] == "similar_images":
            

    return render_template(
        "annotate.html",
        results=similar_images,
        uploaded_image_url=compared_with_img_url,
        number_of_results=number_of_results,
        item=image_uid,
        cold_start=request.method == "GET",
        page_size=PAGE_SIZE,
    )


if __name__ == "__main__":

    app.run(port=8080)
    # from waitress import serve

    # serve(app, host="0.0.0.0", port=8080)
