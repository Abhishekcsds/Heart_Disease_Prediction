"""
Heart Disease Prediction - FastAPI Backend
Provides REST API endpoint for heart disease prediction
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any
import pickle
import numpy as np
import pandas as pd

# ============ HEART DISEASE MODEL CLASS (Must be defined before loading pickle) ============
class HeartDiseaseModel:
    """
    Medical knowledge-based weighted model
    Each feature contributes points based on risk level
    """
    
    def __init__(self):
        # Threshold for classification
        self.threshold = 12
        
        # Define risk points for each feature
        self.risk_points = {
            'Age': {
                (50, 60): 1,
                (60, 70): 2,
                (70, 150): 3
            },
            'Sex': {
                1: 1  # Male
            },
            'Chest pain type': {
                2: 1,   # Atypical angina
                3: 2,   # Non-anginal pain
                4: 3    # Asymptomatic
            },
            'BP': {
                (140, 160): 1,
                (160, 180): 2,
                (180, 300): 3
            },
            'Cholesterol': {
                (240, 300): 1,
                (300, 400): 2,
                (400, 1000): 3
            },
            'FBS over 120': {
                1: 2
            },
            'EKG results': {
                2: 1
            },
            'Max HR': {
                (100, 120): 1,
                (0, 100): 2
            },
            'Exercise angina': {
                1: 3
            },
            'ST depression': {
                (1.5, 2.5): 1,
                (2.5, 4.0): 2,
                (4.0, 20): 3
            },
            'Slope of ST': {
                3: 1
            },
            'Number of vessels fluro': {
                1: 1,
                2: 2,
                3: 3
            },
            'Thallium': {
                6: 1,
                7: 3
            }
        }
    
    def _get_points(self, feature_name, value):
        """Get risk points for a feature value"""
        points = 0
        rule = self.risk_points.get(feature_name, {})
        
        for key, pts in rule.items():
            if isinstance(key, tuple):
                if key[0] <= value < key[1]:
                    points = pts
                    break
            else:
                if value == key:
                    points = pts
                    break
        return points
    
    def calculate_total_points(self, features):
        """Calculate total risk points"""
        total = 0
        
        total += self._get_points('Age', features[0])
        total += self._get_points('Sex', features[1])
        total += self._get_points('Chest pain type', features[2])
        total += self._get_points('BP', features[3])
        total += self._get_points('Cholesterol', features[4])
        total += self._get_points('FBS over 120', features[5])
        total += self._get_points('EKG results', features[6])
        total += self._get_points('Max HR', features[7])
        total += self._get_points('Exercise angina', features[8])
        total += self._get_points('ST depression', features[9])
        total += self._get_points('Slope of ST', features[10])
        total += self._get_points('Number of vessels fluro', features[11])
        total += self._get_points('Thallium', features[12])
        
        return total
    
    def predict(self, X):
        """Predict class (0 or 1)"""
        if len(X.shape) == 2:
            predictions = []
            for i in range(X.shape[0]):
                points = self.calculate_total_points(X[i])
                pred = 1 if points >= self.threshold else 0
                predictions.append(pred)
            return np.array(predictions)
        else:
            points = self.calculate_total_points(X)
            return 1 if points >= self.threshold else 0
    
    def predict_proba(self, X):
        """Get probability scores"""
        if len(X.shape) == 2:
            probas = []
            for i in range(X.shape[0]):
                points = self.calculate_total_points(X[i])
                prob_disease = min(points / 28, 0.95)
                probas.append([1 - prob_disease, prob_disease])
            return np.array(probas)
        else:
            points = self.calculate_total_points(X)
            prob_disease = min(points / 28, 0.95)
            return np.array([1 - prob_disease, prob_disease])


class SimpleScaler:
    """Simple feature scaler - FIXED VERSION"""
    
    def __init__(self):
        self.means = None
        self.stds = None
    
    def fit(self, X):
        """Fit scaler with data"""
        if X is None or len(X) == 0:
            # Default values if no data
            self.means = np.zeros(13)
            self.stds = np.ones(13)
        else:
            self.means = np.mean(X, axis=0)
            self.stds = np.std(X, axis=0)
            self.stds[self.stds == 0] = 1
        return self
    
    def transform(self, X):
        """Transform features"""
        if self.means is None or self.stds is None:
            # If not fitted, return as is
            return X
        return (X - self.means) / self.stds
    
    def fit_transform(self, X):
        """Fit and transform"""
        self.fit(X)
        return self.transform(X)


# ============ FASTAPI APP ============
app = FastAPI(
    title="Heart Disease Prediction API",
    description="API for predicting heart disease based on patient medical data",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Feature columns
feature_columns = ['Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol', 
                   'FBS over 120', 'EKG results', 'Max HR', 'Exercise angina',
                   'ST depression', 'Slope of ST', 'Number of vessels fluro', 'Thallium']

# Load the trained model or create default
model = None
scaler = None

try:
    with open('models/heart_disease_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    model = model_data['model']
    scaler = model_data['scaler']
    
    # ✅ FIX: Check if scaler is properly initialized
    if scaler is None or scaler.means is None:
        print("⚠️ Scaler not fitted, creating default scaler...")
        scaler = SimpleScaler()
        dummy_X = np.zeros((1, len(feature_columns)))
        scaler.fit(dummy_X)
    
    print("✅ Model loaded successfully!")
    print(f"   Features: {len(feature_columns)}")
    print(f"   Scaler ready: {scaler.means is not None}")
    
except FileNotFoundError:
    print("⚠️ Model not found. Creating default model...")
    model = HeartDiseaseModel()
    scaler = SimpleScaler()
    dummy_X = np.zeros((1, len(feature_columns)))
    scaler.fit(dummy_X)
    print("✅ Default model created!")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = HeartDiseaseModel()
    scaler = SimpleScaler()
    dummy_X = np.zeros((1, len(feature_columns)))
    scaler.fit(dummy_X)
    print("✅ Fallback model created!")


# Define request model with validation
class HeartDiseaseInput(BaseModel):
    """Input data schema for heart disease prediction"""
    
    Age: int = Field(..., ge=1, le=120, description="Age in years")
    Sex: int = Field(..., ge=0, le=1, description="Sex: 0=Female, 1=Male")
    Chest_pain_type: int = Field(..., ge=1, le=4, description="Chest pain type: 1-4")
    BP: int = Field(..., ge=50, le=250, description="Resting blood pressure (mm Hg)")
    Cholesterol: int = Field(..., ge=100, le=600, description="Serum cholesterol (mg/dl)")
    FBS_over_120: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    EKG_results: int = Field(..., ge=0, le=2, description="Resting ECG results: 0-2")
    Max_HR: int = Field(..., ge=60, le=220, description="Maximum heart rate achieved")
    Exercise_angina: int = Field(..., ge=0, le=1, description="Exercise induced angina")
    ST_depression: float = Field(..., ge=0.0, le=10.0, description="ST depression induced by exercise")
    Slope_of_ST: int = Field(..., ge=1, le=3, description="Slope of peak exercise ST segment")
    Number_of_vessels_fluro: int = Field(..., ge=0, le=3, description="Number of major vessels colored")
    Thallium: int = Field(..., ge=3, le=7, description="Thallium stress test result")
    
    class Config:
        json_schema_extra = {
            "example": {
                "Age": 58,
                "Sex": 1,
                "Chest_pain_type": 3,
                "BP": 120,
                "Cholesterol": 288,
                "FBS_over_120": 0,
                "EKG_results": 2,
                "Max_HR": 145,
                "Exercise_angina": 1,
                "ST_depression": 0.8,
                "Slope_of_ST": 2,
                "Number_of_vessels_fluro": 3,
                "Thallium": 7
            }
        }


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    probability_no_disease: float
    message: str
    risk_level: str
    confidence: float


@app.get("/")
async def root():
    return {
        "message": "Heart Disease Prediction API",
        "status": "running",
        "model_loaded": model is not None,
        "endpoints": {
            "/predict": "POST - Make prediction",
            "/health": "GET - Check API health",
            "/docs": "GET - API documentation"
        }
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_type": type(model).__name__ if model else None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(data: HeartDiseaseInput):
    """Predict heart disease based on patient medical data"""
    
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure model training has been completed."
        )
    
    try:
        # Convert input to dictionary
        input_dict = data.dict()
        
        # Map input field names to column names
        input_for_model = {
            'Age': input_dict['Age'],
            'Sex': input_dict['Sex'],
            'Chest pain type': input_dict['Chest_pain_type'],
            'BP': input_dict['BP'],
            'Cholesterol': input_dict['Cholesterol'],
            'FBS over 120': input_dict['FBS_over_120'],
            'EKG results': input_dict['EKG_results'],
            'Max HR': input_dict['Max_HR'],
            'Exercise angina': input_dict['Exercise_angina'],
            'ST depression': input_dict['ST_depression'],
            'Slope of ST': input_dict['Slope_of_ST'],
            'Number of vessels fluro': input_dict['Number_of_vessels_fluro'],
            'Thallium': input_dict['Thallium']
        }
        
        # Create array with proper feature ordering
        input_array = np.array([[
            input_for_model['Age'],
            input_for_model['Sex'],
            input_for_model['Chest pain type'],
            input_for_model['BP'],
            input_for_model['Cholesterol'],
            input_for_model['FBS over 120'],
            input_for_model['EKG results'],
            input_for_model['Max HR'],
            input_for_model['Exercise angina'],
            input_for_model['ST depression'],
            input_for_model['Slope of ST'],
            input_for_model['Number of vessels fluro'],
            input_for_model['Thallium']
        ]])
        
        # ✅ FIX: Safe scaling with None check
        if scaler is not None and scaler.means is not None and scaler.stds is not None:
            input_scaled = scaler.transform(input_array)
        else:
            # If scaler not available, use raw values
            input_scaled = input_array
            print("⚠️ Using raw values (scaler not available)")
        
        # Make prediction
        prediction = int(model.predict(input_scaled)[0])
        probabilities = model.predict_proba(input_scaled)[0]
        
        prob_disease = float(probabilities[1])
        prob_no_disease = float(probabilities[0])
        confidence = max(prob_disease, prob_no_disease)
        
        # Determine risk level
        if prob_disease < 0.3:
            risk_level = "Low"
        elif prob_disease < 0.5:
            risk_level = "Moderate"
        elif prob_disease < 0.7:
            risk_level = "High"
        else:
            risk_level = "Very High"
        
        # Create message
        if prediction == 1:
            message = "⚠️ High risk of heart disease detected. Please consult a healthcare provider for further evaluation."
        else:
            message = "✅ Low risk of heart disease. Continue maintaining a healthy lifestyle."
        
        return PredictionResponse(
            prediction=prediction,
            probability=prob_disease,
            probability_no_disease=prob_no_disease,
            message=message,
            risk_level=risk_level,
            confidence=confidence
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/feature-info")
async def get_feature_info():
    return {
        "features": feature_columns,
        "feature_count": len(feature_columns),
        "description": {
            "Age": "Age in years (1-120)",
            "Sex": "Gender (0=Female, 1=Male)",
            "Chest pain type": "Type of chest pain (1-4)",
            "BP": "Resting blood pressure in mm Hg (50-250)",
            "Cholesterol": "Serum cholesterol in mg/dl (100-600)",
            "FBS over 120": "Fasting blood sugar > 120 mg/dl (0=No, 1=Yes)",
            "EKG results": "Resting ECG results (0=Normal, 1=ST-T abnormality, 2=LV hypertrophy)",
            "Max HR": "Maximum heart rate achieved (60-220)",
            "Exercise angina": "Exercise induced angina (0=No, 1=Yes)",
            "ST depression": "ST depression induced by exercise (0-10)",
            "Slope of ST": "Slope of peak exercise ST segment (1-3)",
            "Number of vessels fluro": "Number of major vessels colored (0-3)",
            "Thallium": "Thallium stress test result (3,6,7)"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)