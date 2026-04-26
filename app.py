from flask import Flask, render_template, request
import os
from predict import predict_image
from pymongo import MongoClient
from datetime import datetime

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# MongoDB connection
client = MongoClient("mongodb://localhost:27017/")
db = client["casting_db"]
collection = db["results"]

@app.route("/", methods=["GET", "POST"])
def index():

    result = None
    image_path = None

    if request.method == "POST":

        file = request.files["file"]

        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            result, confidence = predict_image(filepath)
            image_path = filepath

            # Save to MongoDB
            data = {
                "image_name": file.filename,
                "result": result,
                "confidence": float(confidence),
                "timestamp": datetime.now()
            }
            print("Saving to MongoDB:", data)

            collection.insert_one(data)

    return render_template("index.html", result=result, image_path=image_path)

if __name__ == "__main__":
    app.run(debug=True)