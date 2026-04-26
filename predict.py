import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load model once
model = load_model("casting_model.h5")

def predict_image(path):

    img = cv2.imread(path)

    # safety check
    if img is None:
        return "ERROR", 0.0

    # convert BGR → RGB (important)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # resize
    img = cv2.resize(img, (300, 300))

    # normalize
    img = img.astype("float32") / 255.0

    # reshape
    img = np.expand_dims(img, axis=0)

    # prediction
    pred = model.predict(img, verbose=0)[0][0]

    print("Prediction probability:", pred)

    # mapping (VERY IMPORTANT)
    if pred < 0.5:
        return "DEFECT", float(1 - pred)   # confidence for defect
    else:
        return "OK", float(pred)           # confidence for ok