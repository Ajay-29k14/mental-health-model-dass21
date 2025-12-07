cat << 'EOF' > README.md
# Mental Health Prediction using DASS-21 (KIIT Students)

This project builds machine learning models to predict **Depression, Anxiety, and Stress severity levels** from the **DASS-21 mental health questionnaire**.

The models are trained on **real, anonymised responses collected from students of Kalinga Institute of Industrial Technology (KIIT University)** as part of an academic project.

> ⚠️ All data used is anonymised and **not shared publicly** in this repository to preserve the privacy and confidentiality of KIIT students.

---

## 🎯 Project Overview

Given the 21 DASS-21 item scores (each rated from **0 to 3**), the goal is to automatically classify a student's mental health severity into:

- **Normal**
- **Mild**
- **Moderate**
- **Severe**
- **Extremely Severe**

for each of the three subscales:

- **Depression**
- **Anxiety**
- **Stress**

This repository contains the complete implementation of an **end-to-end ML pipeline**.

---

## 📂 Dataset

- **Source:** Real DASS-21 questionnaire responses from **KIIT University students**
- **Features:** 21 item responses (0–3 scale)
- **Targets:** Severity labels

🛡 **Note:** The dataset is private and is **not included** in this repository.

---

## 🛠 Tech Stack

| Category | Tools |
|---------|--------|
| Programming | Python |
| ML Framework | Scikit-Learn |
| Data Handling | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Model Saving | Joblib |

---

## 📊 Model Performance

| Subscale | Accuracy |
|----------|---------|
| **Depression** | **84.09%** |
| **Anxiety** | **76.14%** |
| **Stress** | **77.27%** |

---

## ▶️ How to Run

```bash
# Clone repo
git clone https://github.com/Ajay-29k14/mental-health-model-dass21.git
cd mental-health-model-dass21/src

# Install dependencies
pip install -r ../requirements.txt

# Train models
python model.py

# Example prediction
python predict.py
