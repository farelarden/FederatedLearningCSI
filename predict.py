# predict.py
import pandas as pd
import numpy as np
import torch
import joblib
from model import SimpleRegressionNet

# Load artifacts
scaler = joblib.load("artifacts/scaler.pkl")
imputer = joblib.load("artifacts/imputer.pkl")
model = SimpleRegressionNet(len(imputer.feature_names_in_))
model.load_state_dict(torch.load("artifacts/final_model.pth"))
model.eval()

# Load test data
df = pd.read_csv("x_test.csv", sep=';')
df.columns = df.columns.str.strip().str.lower()
test_ids = df['id'].copy()

# Drop same columns as training
COLS_TO_DROP = {'hr_time_series', 'resp_time_series', 'stress_time_series', 'act_activetime', 'day', 'unnamed: 0', 'id'}
df = df.drop(columns=COLS_TO_DROP, errors='ignore')

# Ensure feature order matches training
X_test = df[imputer.feature_names_in_].values.astype(np.float32)
X_test = imputer.transform(X_test)
X_test = scaler.transform(X_test)

# Predict
with torch.no_grad():
    preds = model(torch.from_numpy(X_test)).numpy().flatten()

# Save submission
pd.DataFrame({'id': test_ids, 'label': preds}).to_csv("submission.csv", index=False)
print("✅ Submission saved!")