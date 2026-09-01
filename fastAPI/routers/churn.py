import joblib
import pandas as pd
from fastapi import APIRouter
from pydantic import BaseModel

# Khởi tạo router thay vì app
router = APIRouter(
    tags=["Churn Prediction"]
)

CHURN_ARTIFACT_PATH = 'data/churnprediction/churn_pipeline.joblib'

try:
    churn_pipeline = joblib.load(CHURN_ARTIFACT_PATH)
except FileNotFoundError:
    churn_pipeline = None
    print(f"Cảnh báo: Chưa tìm thấy file artifact tại {CHURN_ARTIFACT_PATH}")

class ChurnFeatures(BaseModel):
    tenure: float
    age: float
    address: float
    income: float
    ed: float
    employ: float
    equip: float

# Đổi @app.post thành @router.post
@router.post("/churn-prediction")
def predict_churn(customer: ChurnFeatures):
    if churn_pipeline is None:
        return {"error": "Mô hình Churn chưa được tải. Vui lòng chạy file train trước."}
    
    input_df = pd.DataFrame([customer.model_dump()])
    
    prediction = churn_pipeline.predict(input_df)[0]
    probabilities = churn_pipeline.predict_proba(input_df)[0]
    churn_prob = float(probabilities[1])
    
    return {
        "churn_prediction": int(prediction),
        "status": "Churn" if prediction == 1 else "Stay",
        "churn_probability": round(churn_prob, 4),
        "stay_probability": round(1.0 - churn_prob, 4),
        "risk_level": "High" if churn_prob >= 0.6 else ("Medium" if churn_prob >= 0.3 else "Low")
    }