from django.shortcuts import render, redirect
from django.http import JsonResponse
import pandas as pd
import xgboost as xgb
import os

# Load the pre-trained XGBoost model

model = xgb.Booster()
model.load_model('heart_risk_app/ckd_heart_attack_model.model')

def home(request):
    return render(request, 'home.html')


DEFAULT_VALUES = {
    'bgr': 120.0,  
    'bu': 20.0,    
    'sc': 1.2,     
    'sod': 140.0,  
    'pot': 4.5,    
    'wbcc': 7000,  
    'rbcc': 4.5,  
    'htn': 0,      
    'dm': 0,       
    'cad': 0,     
    'appet': 1,    
    'pe': 0,       
    'ane': 0,      
}

def predict_risk(request):
    if request.method == 'POST':
        if model is None:
            return render(request, 'result.html', {
                'risk': 'Error',
                'treatments': ['Model not loaded. Please check the server logs.']
            })

        
        user_data = {
            'age': float(request.POST['age']),
            'bp': float(request.POST['bp']),
            'sg': float(request.POST['sg']),
            'al': float(request.POST['al']),
            'su': float(request.POST['su']),
            'hemo': float(request.POST['hemo']),
            'pcv': float(request.POST['pcv']),
        }

        # Fill missing features with default values
        for feature, default_value in DEFAULT_VALUES.items():
            if feature not in user_data:
                user_data[feature] = default_value

        # Convert the dictionary to a DataFrame
        df = pd.DataFrame([user_data])

        # Ensure the columns are in the correct order (same as the training data)
        df = df[['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']]

        # Convert data to DMatrix (XGBoost's input format)
        dmatrix = xgb.DMatrix(df)

        # Make prediction
        predictions = model.predict(dmatrix)
        risk = "High Risk" if predictions[0] > 0.5 else "Low Risk"

        # Return results
        treatments = {
            "High Risk": [
                "Consult a cardiologist immediately.",
                "Follow a low-sodium diet.",
                "Engage in light exercise regularly."
            ],
            "Low Risk": [
                "Maintain a healthy diet.",
                "Exercise regularly.",
                "Monitor blood pressure and kidney function."
            ]
        }
        return render(request, 'result.html', {
            'risk': risk,
            'treatments': treatments[risk]
        })

    return redirect('home')