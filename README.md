# 🏭 Real-Time Visual Quality Inspection using CNN
## Casting Product Defect Detection System

## 📌 Project Overview

This project is an **AI-powered Visual Quality Inspection System** designed to automatically detect defects in casting products using **Computer Vision and Deep Learning (CNN)**.

The system allows users to upload an image of a casting product and instantly receive a prediction:

* ✅ **OK Product**
* ❌ **Defective Product**

It also stores inspection results in a database for tracking and analysis.

---

## 🚀 Features

* 📷 Upload casting product image
* 🤖 AI-based defect detection using CNN
* ⚡ Real-time prediction
* 🧠 Trained on industrial dataset (Kaggle)
* 💾 Stores results in MongoDB
* 🌐 Web interface using Flask
* 📊 (Optional) Prediction history tracking

---

## 🏗️ Project Architecture

```
User Upload Image
        ↓
Flask Web App (Frontend + Backend)
        ↓
CNN Model (casting_model.h5)
        ↓
Prediction (OK / DEFECT)
        ↓
MongoDB (Store Results)
```

---

## 🧠 Model Training (Deep Learning)

### 📂 Dataset

* Source: Kaggle (Casting Product Image Dataset)
* Classes:

  * `ok_front`
  * `def_front`

### ⚙️ Preprocessing

* Resize images → `(300, 300)`
* Normalize pixel values → `[0,1]`
* ImageDataGenerator used

### 🧩 Model Architecture

* Convolutional Neural Network (CNN)
* Layers:

  * Conv2D + ReLU
  * MaxPooling
  * Flatten
  * Dense Layers
  * Sigmoid Output

### 📊 Training Details

* Epochs: 10
* Accuracy: ~99%
* Loss: Binary Crossentropy
* Optimizer: Adam

### 💾 Model Output

```
casting_model.h5
```

---

## 🖥️ Backend (Flask)

### 📌 Features

* Handles image upload
* Calls prediction function
* Saves results to MongoDB

### 📄 Main File: `app.py`

## 🤖 Prediction Logic: `predict.py`


## 🌐 Frontend

### 📄 File: `templates/index.html`

* Built using HTML + CSS
* Features:

  * Upload image
  * Display result
  * Show confidence

---

## 🗄️ Database (MongoDB)

* Database Name: `casting_db`
* Collection: `results`

### 📌 Stored Data Example

```json
{
  "image_name": "x1.jpg",
  "result": "OK",
  "confidence": 0.98,
  "timestamp": "2026-04-26"
}
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```
git clone <repo-url>
cd project-folder
```

### 2️⃣ Create Virtual Environment

```
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```
pip install flask tensorflow opencv-python numpy pymongo
```

### 4️⃣ Run MongoDB

Make sure MongoDB is running:

```
mongod
```

### 5️⃣ Run Application

```
python app.py
```

### 6️⃣ Open Browser

```
http://127.0.0.1:5000
```

---

## 📊 Results

* High accuracy (~99%)
* Fast predictions
* Works on real casting images

---

## 🔮 Future Improvements

* 📊 Dashboard (Defect vs OK analytics)
* 🔍 Defect localization (Grad-CAM)
* ☁️ Deployment on cloud
* 📱 Mobile app integration

---

## 👨‍💻 Author

**Mahek Aggarwal**
B.Tech AI & Data Science

---

