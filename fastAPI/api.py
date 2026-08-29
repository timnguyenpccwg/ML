from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="CO2 Emission Prediction API")

# Load artifacts from storage
poly = joblib.load('data/co2prediction/poly.joblib')
scaler = joblib.load('data/co2prediction/scaler.joblib')
model = joblib.load('data/co2prediction/model.joblib')

class CarFeatures(BaseModel):
    ENGINESIZE: float
    FUELCONSUMPTION_COMB_MPG: float

@app.post("/co2predict")
def predict_co2(car: CarFeatures):
    # Convert input request to Pandas DataFrame
    new_car_df = pd.DataFrame([car.model_dump()])
    features = ['ENGINESIZE', 'FUELCONSUMPTION_COMB_MPG']
    
    # Extract features array in expected order
    new_car_array = new_car_df[features].to_numpy()
    
    # Transform pipeline: Polynomial -> Scaler -> Model
    new_car_poly = poly.transform(new_car_array)
    new_car_std = scaler.transform(new_car_poly)
    co2_prediction = model.predict(new_car_std)
    
    return {
        "engine_size": car.ENGINESIZE,
        "mpg": car.FUELCONSUMPTION_COMB_MPG,
        "predicted_co2_g_km": round(co2_prediction.item(), 2)
    }