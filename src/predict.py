import joblib
import numpy as np

# Load models
model_dep = joblib.load("../models/rf_model_depression.pkl")
model_anx = joblib.load("../models/rf_model_anxiety.pkl")
model_str = joblib.load("../models/rf_model_stress.pkl")


sample_response = [
    1, 0, 0, 0, 0, 1, 0,
    1, 3, 0, 0, 0, 1, 3,
    0, 2, 2, 1, 0, 1, 0
]

X_new = np.array(sample_response).reshape(1, -1)

dep_pred = model_dep.predict(X_new)[0]
anx_pred = model_anx.predict(X_new)[0]
str_pred = model_str.predict(X_new)[0]

print("Predicted Depression Severity:", dep_pred)
print("Predicted Anxiety Severity   :", anx_pred)
print("Predicted Stress Severity    :", str_pred)
