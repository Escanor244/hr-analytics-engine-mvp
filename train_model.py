# train_model.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import os

# Load data
df = pd.read_csv('data/hr_data.csv')

# Drop unnecessary columns
df.drop(['EmployeeNumber', 'Over18', 'StandardHours', 'EmployeeCount'], axis=1, inplace=True)

# Encode target
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# Encode categoricals
categorical_cols = df.select_dtypes(include='object').columns
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Features and label
X = df.drop('Attrition', axis=1)
y = df['Attrition']

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train model with class balancing
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=(len(y) - sum(y)) / sum(y))
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.2f}")

# Save model, encoders, and column order
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/attrition_model.pkl')
joblib.dump(encoders, 'models/label_encoders.pkl')
joblib.dump(X.columns.tolist(), 'models/input_columns.pkl')  # Save column order

print("✅ Model, encoders, and columns saved.")
