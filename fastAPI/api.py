from fastapi import FastAPI
from pydantic import BaseModel
from routers import churn
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI(title="Multi-Model Prediction Platform")

app.include_router(churn.router)

# ==========================================
# 1. KHAI BÁO MODEL 1: POLYNOMIAL SCIKIT-LEARN
# ==========================================
CO2_ARTIFACTS_DIR = 'data/co2prediction'
poly_co2 = joblib.load(os.path.join(CO2_ARTIFACTS_DIR, 'poly.joblib'))
scaler_co2 = joblib.load(os.path.join(CO2_ARTIFACTS_DIR, 'scaler.joblib'))
model_co2 = joblib.load(os.path.join(CO2_ARTIFACTS_DIR, 'model.joblib'))

class CO2Features(BaseModel):
    ENGINESIZE: float
    FUELCONSUMPTION_COMB_MPG: float

@app.post("/co2prediction")
def predict_co2(car: CO2Features):
    input_df = pd.DataFrame([car.model_dump()])
    features = ['ENGINESIZE', 'FUELCONSUMPTION_COMB_MPG']
    
    raw_features = input_df[features].to_numpy()
    poly_features = poly_co2.transform(raw_features)
    scaled_features = scaler_co2.transform(poly_features)
    prediction = model_co2.predict(scaled_features)
    
    return {
        "engine_size": car.ENGINESIZE,
        "mpg": car.FUELCONSUMPTION_COMB_MPG,
        "predicted_co2_g_km": round(prediction.item(), 2)
    }

# ==========================================
# 2. KHAI BÁO MODEL 2: CUSTOM GRADIENT DESCENT
# ==========================================
# Cập nhật đường dẫn theo đúng cấu trúc thư mục trong ảnh
GD_ARTIFACTS_DIR = 'data/gd-co2prediction'

try:
    gd_theta = np.load(os.path.join(GD_ARTIFACTS_DIR, 'gd_theta.npy'))
    gd_mean = np.load(os.path.join(GD_ARTIFACTS_DIR, 'gd_mean.npy'))
    gd_std = np.load(os.path.join(GD_ARTIFACTS_DIR, 'gd_std.npy'))
except FileNotFoundError:
    print("Cảnh báo: Không tìm thấy file npy, đang sử dụng giá trị mặc định.")
    gd_theta = np.array([[256.22], [15.36], [13.50], [33.43]])
    gd_mean = np.array([3.34, 5.79, 11.58])
    gd_std = np.array([1.41, 1.79, 3.48])

class GDCO2Features(BaseModel):
    ENGINESIZE: float
    CYLINDERS: int
    FUELCONSUMPTION_COMB: float

@app.post("/gd-co2-prediction")
def predict_gd_co2(car: GDCO2Features):
    # 1. Trích xuất mảng dữ liệu (Raw Data)
    X_raw = np.array([[car.ENGINESIZE, car.CYLINDERS, car.FUELCONSUMPTION_COMB]])
    
    # 2. Chuẩn hóa Z-Score (Scale bằng Mean và Std)
    X_scaled = (X_raw - gd_mean) / gd_std
    
    # 3. Chèn cột Bias (số 1) vào đầu tiên
    X_b = np.c_[np.ones((1, 1)), X_scaled]
    
    # 4. Dự đoán (Nhân Ma trận: X_b dot Theta)
    prediction = np.dot(X_b, gd_theta)
    
    return {
        "model_used": "Custom Gradient Descent",
        "inputs": car.model_dump(),
        "predicted_co2_g_km": round(prediction.item(), 2)
    }