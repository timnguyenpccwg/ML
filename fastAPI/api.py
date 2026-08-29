from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="CO2 Emission Prediction API")

# Đánh thức mô hình sẵn sàng trên RAM
scaler = joblib.load('data/scaler.joblib')
model = joblib.load('data/model.joblib')

# Định nghĩa cấu trúc dữ liệu đầu vào chặt chẽ (Data Validation)
class CarFeatures(BaseModel):
    ENGINESIZE: float
    FUELCONSUMPTION_COMB_MPG: float

@app.post("/predict")
def predict_co2(car: CarFeatures):
    # Chuyển đổi JSON input thành DataFrame
    new_car_df = pd.DataFrame([car.model_dump()])
    
    # Trích xuất và chuẩn hóa
    features = ['ENGINESIZE', 'FUELCONSUMPTION_COMB_MPG']
    new_car_array = new_car_df[features].to_numpy()
    new_car_std = scaler.transform(new_car_array)
    
    # Dự đoán
    co2_prediction = model.predict(new_car_std)
    
    # Trả về kết quả JSON
    return {
        "ENGINESIZE": car.ENGINESIZE,
        "FUELCONSUMPTION_COMB_MPG": car.FUELCONSUMPTION_COMB_MPG,
        "predicted_co2_g_km": round(co2_prediction.item(), 2)
    }