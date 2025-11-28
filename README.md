# 🧠 Brain Tumor Detection Web App (Streamlit)

A deep learning–powered web application that detects brain tumors from MRI images using a **MobileNet-based CNN model**, with powerful explainability features using **Grad-CAM** and **LIME**.  
The project is deployed using **Streamlit**, providing an easy-to-use and interactive interface.

---

## 📌 Features

- ✔️ Upload MRI images for **tumor classification**  
- ✔️ Uses a trained **MobileNet** deep learning model  
- ✔️ **Grad-CAM heatmap** to show which regions contribute most  
- ✔️ **LIME explanation** for interpretability  
- ✔️ Clean Streamlit interface  
- ✔️ Fast, lightweight, and easy to deploy  

---

## 🧑‍⚕️ Dataset

This project uses the **Brain Tumor MRI Dataset** from Kaggle:

🔗 **Dataset Link:**  
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data

The dataset contains 4 MRI classes:

- **Glioma Tumor**
- **Meningioma Tumor**
- **Pituitary Tumor**
- **No Tumor**

---

## 🧠 Model Information

- Architecture: **MobileNet (Transfer Learning)**
- Input Size: 224×224×3  
- Optimizer: Adam  
- Loss Function: Categorical Crossentropy  
- Output Classes: 4  
- Training Framework: TensorFlow/Keras  

Trained model included in this repo:

---

## 🔍 Explainability (XAI)

### 📕 Grad-CAM  
Generates heatmaps to highlight tumor regions that influence predictions.  
Implemented in:

### 📗 LIME  
Produces superpixel-based explanations for model transparency.  
Implemented in:



