from flask import Flask, render_template, request
import os
from predict import predict_image

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# create upload folder if not exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():

    result = None
    image_path = None

    if request.method == "POST":

        file = request.files["file"]

        if file:

            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            result = predict_image(filepath)
            image_path = filepath

    return render_template("index.html", result=result, image_path=image_path)


if __name__ == "__main__":
    app.run(debug=True)