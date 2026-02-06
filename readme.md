# 🫁 LungAI – Clinical Chest X-Ray Analysis System

LungAI is an **AI-powered clinical decision support system** designed to analyze chest X-ray images using deep learning.  
The system classifies X-ray images into **Normal**, **Lung Opacity**, or **Viral Pneumonia**, providing probability-based predictions along with clinical guidance.

> ⚠️ This application is intended for **educational and research purposes only** and must not be used as a standalone medical diagnostic tool.

---

## 🚀 Live Demo
👉 (https://lungaixray.streamlit.app/)

## 🚀 Key Features

- > Deep Learning–based Chest X-ray Classification
- > ResNet50 (Transfer Learning)
- > Clean & Professional Clinical UI (Streamlit)
- > Probability-based Confidence Scores
- > AI-generated Clinical Insights & Next-Step Suggestions
- > Industry-safe Medical Disclaimer
- > Deploy-ready Application

---

## 🧠 Model Overview

| Component | Details |
|--------|--------|
| Architecture | ResNet50 |
| Framework | PyTorch |
| Classes | Normal, Lung Opacity, Viral Pneumonia |
| Input | Chest X-ray Images |
| Output | Class Probabilities |
| Mode | Inference Only |

---

## 📂 Dataset

The model was trained using publicly available chest X-ray datasets.

### 🔗 Kaggle Dataset Link:
**Chest X-Ray Images (Pneumonia)**  
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

**Classes Used:**
- Normal
- Lung Opacity
- Viral Pneumonia

---

## 🏗️ Project Structure

```text
X_Ray_Project_DL/
│
├── app.py                  # Streamlit application
├── config.py               # Configuration (IMG_SIZE, DEVICE, etc.)
├── requirements.txt        # Dependencies
│
├── model/
│   └── best_model.pth      # Trained PyTorch model
│
├── src/
│   └── model.py            # ResNet50 model definition
│
└── README.md
