import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Step 1: Create a Synthetic Dataset
data = {
    'age': [50, 60, 45, 55, 65, 70, 40, 48, 52, 58],
    'bp': [80, 90, 85, 75, 95, 100, 70, 78, 88, 82],
    'sg': [1.020, 1.015, 1.010, 1.025, 1.030, 1.018, 1.022, 1.016, 1.019, 1.024],
    'al': [0, 1, 0, 2, 1, 0, 1, 0, 2, 1],
    'su': [0, 0, 1, 0, 1, 0, 0, 1, 0, 1],
    'bgr': [120, 140, 130, 110, 150, 160, 100, 125, 135, 145],
    'bu': [20, 25, 18, 22, 30, 35, 15, 19, 24, 28],
    'sc': [1.2, 1.5, 1.0, 1.3, 1.6, 1.7, 0.9, 1.1, 1.4, 1.55],
    'sod': [140, 135, 142, 138, 130, 128, 145, 139, 136, 132],
    'pot': [4.5, 4.8, 4.2, 4.6, 5.0, 5.2, 4.0, 4.3, 4.7, 4.9],
    'hemo': [12.5, 11.8, 13.0, 12.0, 11.0, 10.5, 13.5, 12.8, 11.5, 11.2],
    'pcv': [38, 36, 40, 37, 35, 34, 41, 39, 36, 35],
    'wbcc': [7000, 7500, 6800, 7200, 8000, 8500, 6500, 7100, 7400, 7800],
    'rbcc': [4.5, 4.2, 4.8, 4.6, 4.0, 3.8, 5.0, 4.7, 4.3, 4.1],
    'htn': [1, 1, 0, 1, 1, 1, 0, 1, 1, 1],  # Hypertension (1 = Yes, 0 = No)
    'dm': [1, 0, 1, 0, 1, 1, 0, 1, 0, 1],  # Diabetes Mellitus (1 = Yes, 0 = No)
    'cad': [0, 1, 0, 1, 1, 1, 0, 0, 1, 1],  # Coronary Artery Disease (1 = Yes, 0 = No)
    'appet': [1, 0, 1, 1, 0, 0, 1, 1, 0, 1],  # Appetite (1 = Good, 0 = Poor)
    'pe': [0, 1, 0, 1, 1, 1, 0, 0, 1, 1],  # Pedal Edema (1 = Yes, 0 = No)
    'ane': [0, 1, 0, 1, 1, 1, 0, 0, 1, 1],  # Anemia (1 = Yes, 0 = No)
    'class': [0, 1, 0, 1, 1, 1, 0, 0, 1, 1]  # Target (1 = High Risk, 0 = Low Risk)
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Step 2: Split Data into Features and Target
X = df.drop('class', axis=1)  # Features
y = df['class']  # Target

# Step 3: Train XGBoost Model
model = xgb.XGBClassifier(
    objective='binary:logistic',  # Binary classification
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

model.fit(X, y)

# Step 4: Save the Model
model.save_model('ckd_heart_attack_model.model')
print("Model saved as 'ckd_heart_attack_model.model'")