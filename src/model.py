import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# 1. Load dataset
file_path = "../data/kiit_dass21_clean.xlsx"   # path from src/ to data/

df = pd.read_excel(file_path)

# Keep only first 21 columns (questions)
df = df.iloc[:, :21]

# Ensure all values are numeric
df = df.apply(pd.to_numeric, errors='raise')

question_cols = df.columns.tolist()

# 2. Compute DASS-21 scores

depression_items = [3, 5, 10, 13, 16, 17, 21]
anxiety_items    = [2, 4, 7, 9, 15, 19, 20]
stress_items     = [1, 6, 8, 11, 12, 14, 18]

dep_idx = [i - 1 for i in depression_items]
anx_idx = [i - 1 for i in anxiety_items]
str_idx = [i - 1 for i in stress_items]

df["Depression_Score"] = df.iloc[:, dep_idx].sum(axis=1) * 2
df["Anxiety_Score"]    = df.iloc[:, anx_idx].sum(axis=1) * 2
df["Stress_Score"]     = df.iloc[:, str_idx].sum(axis=1) * 2

# 3. Create severity labels

def label_depression(x):
    if x <= 9: return "Normal"
    elif x <= 13: return "Mild"
    elif x <= 20: return "Moderate"
    elif x <= 27: return "Severe"
    else: return "Extremely Severe"

def label_anxiety(x):
    if x <= 7: return "Normal"
    elif x <= 9: return "Mild"
    elif x <= 14: return "Moderate"
    elif x <= 19: return "Severe"
    else: return "Extremely Severe"

def label_stress(x):
    if x <= 14: return "Normal"
    elif x <= 18: return "Mild"
    elif x <= 25: return "Moderate"
    elif x <= 33: return "Severe"
    else: return "Extremely Severe"

df["Depression_Label"] = df["Depression_Score"].apply(label_depression)
df["Anxiety_Label"]    = df["Anxiety_Score"].apply(label_anxiety)
df["Stress_Label"]     = df["Stress_Score"].apply(label_stress)

targets = {
    "Depression_Label": "Depression",
    "Anxiety_Label": "Anxiety",
    "Stress_Label": "Stress"
}

X = df[question_cols]

results = {}

for target_col, name in targets.items():
    print("\n========================")
    print(f" Training for {name} ")
    print("========================")

    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n{name} Accuracy: {acc:.4f}\n")
    print("Classification report:")
    print(classification_report(y_test, y_pred))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    results[name] = {"model": model, "accuracy": acc}

    # Save model
    joblib.dump(model, f"../rf_model_{name.lower()}.pkl")
    print(f"Saved model: rf_model_{name.lower()}.pkl")

print("\nAll models trained and saved.")
