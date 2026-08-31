import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# --- 1. CẤU HÌNH ---
DATA_PATH = 'data/FuelConsumptionCo2.csv'
ARTIFACTS_DIR = 'data/co2prediction'
TEST_SIZE = 0.2
RANDOM_STATE = 42

def train_sklearn_pipeline(df):
    print("--- BẮT ĐẦU HUẤN LUYỆN: SCIKIT-LEARN PIPELINE ---")
    X = df[['ENGINESIZE', 'FUELCONSUMPTION_COMB_MPG']]
    y = df['CO2EMISSIONS']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    print(f"[Scikit-Learn] R2 Score: {r2_score(y_test, y_pred) * 100:.2f}%")
    
    # Export File
    joblib.dump(pipeline, os.path.join(ARTIFACTS_DIR, 'co2_pipeline.joblib'))
    print("[Scikit-Learn] Đã lưu: co2_pipeline.joblib\n")

def train_custom_gradient_descent(df):
    print("--- BẮT ĐẦU HUẤN LUYỆN: CUSTOM GRADIENT DESCENT ---")
    # Lấy 3 đặc trưng như thiết kế trên API
    X_raw = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']].values
    y = df['CO2EMISSIONS'].values.reshape(-1, 1)

    m = len(y)

    # Chuẩn hóa Z-Score (tự code)
    X_mean = np.mean(X_raw, axis=0)
    X_std = np.std(X_raw, axis=0)
    X_scaled = (X_raw - X_mean) / X_std

    # Thêm cột Bias (toàn số 1)
    X_b = np.c_[np.ones((m, 1)), X_scaled]

    # Vòng lặp Gradient Descent
    theta = np.zeros((4, 1))
    learning_rate = 0.1
    epochs = 1000

    for i in range(epochs):
        y_pred = np.dot(X_b, theta)
        error = y_pred - y
        gradient = (1/m) * np.dot(X_b.T, error)
        theta = theta - learning_rate * gradient

    # Đánh giá RMSE
    y_final = np.dot(X_b, theta)
    rmse = np.sqrt(np.mean((y_final - y)**2))
    print(f"[Gradient Descent] Sai số RMSE: {rmse:.2f} g/km")

    # Export Files
    np.save(os.path.join(ARTIFACTS_DIR, 'gd_theta.npy'), theta)
    np.save(os.path.join(ARTIFACTS_DIR, 'gd_mean.npy'), X_mean)
    np.save(os.path.join(ARTIFACTS_DIR, 'gd_std.npy'), X_std)
    print("[Gradient Descent] Đã lưu: gd_theta.npy, gd_mean.npy, gd_std.npy\n")

def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    
    # Chạy song song 2 tiến trình huấn luyện
    train_sklearn_pipeline(df)
    train_custom_gradient_descent(df)
    
    print("HOÀN TẤT HUẤN LUYỆN VÀ EXPORT TOÀN BỘ MÔ HÌNH!")

if __name__ == "__main__":
    main()