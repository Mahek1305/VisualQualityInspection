import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("casting_model.h5")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    img = cv2.resize(frame, (128,128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    if prediction[0][0] > 0.5:
        label = "OK"
        color = (0,255,0)
    else:
        label = "DEFECT"
        color = (0,0,255)

    cv2.putText(frame, label, (30,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, color, 2)

    cv2.imshow("Casting Quality Inspection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()