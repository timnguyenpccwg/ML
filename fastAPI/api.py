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
    xe_moi_df = pd.DataFrame([car.model_dump()])
    
    # Trích xuất và chuẩn hóa
    features = ['ENGINESIZE', 'FUELCONSUMPTION_COMB_MPG']
    xe_moi_array = xe_moi_df[features].to_numpy()
    xe_moi_std = scaler.transform(xe_moi_array)
    
    # Dự đoán
    co2_du_doan = model.predict(xe_moi_std)
    
    # Trả về kết quả JSON
    return {
        "engine_size": car.ENGINESIZE,
        "mpg": car.FUELCONSUMPTION_COMB_MPG,
        "predicted_co2_g_km": round(co2_du_doan.item(), 2)
    }