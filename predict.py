import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("casting_model.h5")

def predict_image(path):

    img = cv2.imread(path)

    # convert BGR → RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # resize to model input
    img = cv2.resize(img, (300,300))

    # normalize
    img = img.astype("float32") / 255.0

    # reshape
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0][0]

    print("Prediction probability:", prediction)

    # class mapping
    if prediction < 0.5:
        return "DEFECT"
    else:
        return "OK"