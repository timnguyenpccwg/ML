from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(title="Multi-Model Prediction Platform")

# Load CO2 Model Artifacts
CO2_ARTIFACTS_DIR = 'data/co2prediction'
poly_co2 = joblib.load(os.path.join(CO2_ARTIFACTS_DIR, 'poly.joblib'))
scaler_co2 = joblib.load(os.path.join(CO2_ARTIFACTS_DIR, 'scaler.joblib'))
model_co2 = joblib.load(os.path.join(CO2_ARTIFACTS_DIR, 'model.joblib'))

class CO2Features(BaseModel):
    ENGINESIZE: float
    FUELCONSUMPTION_COMB_MPG: float

@app.post("/co2prediction")
def predict_co2(car: CO2Features):
    # Convert payload to DataFrame
    input_df = pd.DataFrame([car.model_dump()])
    features = ['ENGINESIZE', 'FUELCONSUMPTION_COMB_MPG']
    
    # Feature transformations
    raw_features = input_df[features].to_numpy()
    poly_features = poly_co2.transform(raw_features)
    scaled_features = scaler_co2.transform(poly_features)
    
    # Inference
    prediction = model_co2.predict(scaled_features)
    
    return {
        "engine_size": car.ENGINESIZE,
        "mpg": car.FUELCONSUMPTION_COMB_MPG,
        "predicted_co2_g_km": round(prediction.item(), 2)
    }