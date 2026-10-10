from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os

from transform_engine import process_image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)

UPLOAD_DIR = os.path.join(BASE_DIR, "input")
RESULT_DIR = os.path.join(BASE_DIR, "static", "results")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No image selected."}), 400

    filename = secure_filename(file.filename)
    input_path = os.path.join(UPLOAD_DIR, filename)

    file.save(input_path)

    try:
        results = process_image(
            input_path,
            RESULT_DIR
        )

        results["images"] = {
            "grayscale": "/static/results/grayscale.png",
            "dft_50": "/static/results/dft_50.png",
            "dft_25": "/static/results/dft_25.png",
            "dct_50": "/static/results/dct_50.png",
            "dct_25": "/static/results/dct_25.png"
        }

        return jsonify(results)

    except Exception as error:
        print("PROCESSING ERROR:", error)

        return jsonify({
            "error": str(error)
        }), 500


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )