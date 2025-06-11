# utils.py

import pandas as pd

def estimate_burnout(row):
    """
    Estimate burnout risk based on several rules of thumb.
    You can customize this with more advanced logic later.
    """
    score = 0
    if row['OverTime'] == 'Yes': score += 1
    if row['JobSatisfaction'] <= 2: score += 1
    if row['WorkLifeBalance'] <= 2: score += 1
    if row['TotalWorkingYears'] >= 10 and row['YearsSinceLastPromotion'] > 3: score += 1

    if score >= 3:
        return 'High'
    elif score == 2:
        return 'Medium'
    else:
        return 'Low'

def preprocess_input(df, encoders):
    """
    Encode categorical columns using saved label encoders.
    """
    for col in encoders:
        if col in df.columns:
            df[col] = encoders[col].transform(df[col])
    return df

def load_model_and_encoders(model_path='models/attrition_model.pkl', encoders_path='models/label_encoders.pkl'):
    import joblib
    model = joblib.load(model_path)
    encoders = joblib.load(encoders_path)
    return model, encoders