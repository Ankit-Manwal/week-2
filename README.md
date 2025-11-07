# 🌿 Plant Disease Detection using CNN for Sustainable Agriculture

## 🧾 Project Overview
This project focuses on detecting **plant leaf diseases** using **Convolutional Neural Networks (CNNs)**.  
It helps farmers identify diseases in crops like **Tomato**, **Potato**, and **Pepper** through image-based analysis, supporting **sustainable agriculture** and improved crop management.

---

## ❗ Problem Statement
Agriculture is one of the most vital sectors globally, yet it faces huge losses due to plant diseases.  
Manual inspection for disease detection is often:
- Time-consuming  
- Inaccurate  
- Dependent on expert knowledge  

Farmers often fail to detect diseases early, leading to:
- Reduced yield  
- Unnecessary pesticide use  
- Lower income  

Hence, there is a strong need for an **automated, accurate, and efficient** plant disease detection system using deep learning.

---

## 💡 Proposed Solution
We propose a **CNN-based deep learning model** capable of identifying plant leaf diseases automatically from images.  

### The proposed system:
- Accepts an image of a crop leaf  
- Processes it through a CNN model  
- Classifies the image into categories (e.g., *Tomato Early Blight*, *Potato Late Blight*, *Healthy*, etc.)  
- Displays the predicted disease and confidence level  

This solution helps farmers detect diseases early, take preventive actions, and reduce chemical usage — promoting **eco-friendly and sustainable agriculture**.

---

## 🎯 Objectives
- Build a CNN model for automatic plant disease detection  
- Train the model using the **PlantVillage dataset**  
- Evaluate the model using accuracy and validation metrics  
- Develop a simple interface for user interaction (via Flask/Streamlit)

---

## 🧩 Methodology

### 1️⃣ Data Collection
Dataset: [PlantVillage Dataset (Kaggle)](https://www.kaggle.com/datasets/emmarex/plantdisease)  
Crops covered include:
- Tomato  
- Potato  
- Pepper  

Each has healthy and diseased leaf images.

---

### 2️⃣ Data Preprocessing
- Image resizing (128×128 pixels)  
- Normalization  
- Data Augmentation (rotation, zoom, flip, etc.)  

---

### 3️⃣ Model Architecture
A Convolutional Neural Network (CNN) is used:
1. **Input Layer:** 128x128 RGB image  
2. **Conv2D + MaxPooling Layers:** Feature extraction  
3. **Dropout Layers:** Prevent overfitting  
4. **Dense Layers:** Classification  
5. **Output Layer:** Softmax activation  

---

### 4️⃣ Training Parameters
- Optimizer: `Adam`  
- Loss Function: `Categorical Crossentropy`  
- Metrics: `Accuracy`  
- Epochs: 20–30 (configurable)  

---

## ⚙️ System Requirements

### 🧠 Software
- Python 3.8+  
- TensorFlow / Keras  
- OpenCV  
- NumPy, Pandas  
- Matplotlib / Seaborn  
- Streamlit or Flask (for web app)

### 💻 Hardware
- GPU recommended for faster training (optional)

