# 🏭 Real-Time Visual Quality Inspection using CNN
## Casting Product Defect Detection System

---

## 📌 Overview

This project implements a Real-Time Visual Quality Inspection System for detecting defective metal casting products using a Convolutional Neural Network (CNN).

The system is trained on a casting image dataset containing two classes:

- OK Casting
- Defective Casting

After training, the model is deployed using OpenCV to perform real-time inspection through a webcam, simulating a small production line setup.

---

## 🎯 Objective

The objective of this project is to:

- Automate visual inspection of casting products
- Detect defective products accurately
- Reduce human inspection effort
- Simulate real-time production line quality control

---

## 🧠 Technologies Used

- Python 3.x
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- Convolutional Neural Network (CNN)

---

## 📂 Dataset Structure

The dataset should be organized as follows:

casting_data/
│
├── train/
│   ├── def_front/
│   └── ok_front/
│
└── test/
    ├── def_front/
    └── ok_front/

- def_front → Defective casting images
- ok_front → Good casting images

This is a Binary Image Classification problem.

---

## 🏗 Model Architecture

The CNN model consists of:

1. Conv2D (32 filters, ReLU activation)
2. MaxPooling2D
3. Conv2D (64 filters, ReLU activation)
4. MaxPooling2D
5. Conv2D (128 filters, ReLU activation)
6. MaxPooling2D
7. Flatten Layer
8. Dense Layer (128 units, ReLU)
9. Dropout (0.5)
10. Output Layer (1 unit, Sigmoid activation)

Model Configuration:

- Optimizer: Adam
- Loss Function: Binary Crossentropy
- Metrics: Accuracy
- Image Size: 128 × 128
- Epochs: 10

---

## 🚀 How to Run the Project

### 1️⃣ Install Dependencies

Open terminal and run:

pip install tensorflow opencv-python numpy matplotlib

---

### 2️⃣ Train the Model

Make sure your dataset is placed inside:

casting_data/train  
casting_data/test  

Run:

python train.py

This will:
- Load and preprocess images
- Train the CNN model
- Save the trained model as casting_model.h5

---

### 3️⃣ Run Real-Time Inspection

After training, run:

python realtime.py

The webcam will open and display:

- OK (Green)
- DEFECT (Red)

Press Q to exit the application.

---

## ⚙️ Working Principle

1. Capture frame from camera
2. Resize frame to 128×128
3. Normalize pixel values
4. Pass image to trained CNN model
5. Predict class (OK / DEFECT)
6. Display result on screen
7. Repeat continuously

---

## 📊 Results

- Successfully classifies casting products
- Provides real-time prediction using webcam
- Achieves high accuracy on test dataset

---

## 🏭 Industrial Applications

- Small-scale production lines
- Metal casting factories
- Automated quality inspection
- Manufacturing defect detection systems

---

## 🔮 Future Improvements

- Implement Transfer Learning (MobileNet, EfficientNet)
- Add defect localization using YOLO
- Add confidence score display
- Save defective images automatically
- Deploy on embedded systems (Raspberry Pi / Jetson Nano)

---

## 📌 Conclusion

This project demonstrates the application of Deep Learning in industrial automation.  
A CNN-based model is trained to classify casting defects and deployed for real-time visual inspection using a webcam.

The system simulates a small production line quality control environment and showcases how AI can enhance manufacturing processes.

---

## 👨‍💻 Author
Mahek, Palak & Anchal
B.Tech. AI & DS
CGC University, Mohali
