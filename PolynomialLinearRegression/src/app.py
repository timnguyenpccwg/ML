import os
import joblib
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# 1. Khai báo Hằng số Cấu hình (Configuration Constants)
DATA_PATH = 'data/FuelConsumptionCo2.csv'
FEATURES = ['ENGINESIZE', 'FUELCONSUMPTION_COMB_MPG']
TARGET = 'CO2EMISSIONS'
POLY_DEGREE = 2
TEST_SIZE = 0.2
RANDOM_STATE = 42

# 2. Nạp Dữ liệu
df = pd.read_csv(DATA_PATH)
X = df[FEATURES]
y = df[TARGET]

# 3. Tạo Đặc trưng Đa thức (Polynomial Features Transformation)
poly = PolynomialFeatures(degree=POLY_DEGREE, include_bias=False)
X_poly = poly.fit_transform(X)

# Lấy danh sách tên đặc trưng mới (ví dụ: ENGINESIZE^2, ENGINESIZE * MPG, ...)
poly_feature_names = poly.get_feature_names_out(FEATURES)

# 4. Chuẩn hóa Dữ liệu (Standardization)
std_scaler = StandardScaler()
X_std = std_scaler.fit_transform(X_poly)

# 5. Chia tách Dữ liệu Train / Test
X_train, X_test, y_train, y_test = train_test_split(
    X_std, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# 6. Huấn luyện Mô hình
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# 7. Đánh giá Mô hình trên Tập Test
y_pred = regressor.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("=" * 60)
print(f"BÁO CÁO ĐÁNH GIÁ MÔ HÌNH POLYNOMIAL REGRESSION (DEGREE = {POLY_DEGREE})")
print("=" * 60)
print(f"Hệ số xác định R2 Score : {r2 * 100:.2f}%")
print(f"Sai số Mean Squared Error: {mse:.2f}")
print("=" * 60)

# 8. Giải mã Hệ số Toán học Tường minh (Không Magic Numbers)
c_intercept = regressor.intercept_.item()

# Chuyển đổi hệ số về không gian gốc để in phương trình
coef_original = regressor.coef_ / std_scaler.scale_
coef_dict = dict(zip(poly_feature_names, coef_original))

print("\n[Phương trình Hồi quy Đa thức gốc]:")
equation_terms = [f"{c_intercept:.2f}"]
for feature, coef in coef_dict.items():
    equation_terms.append(f"({coef:.2f} * {feature})")
print(f"CO2 = {' + '.join(equation_terms)}")
print("=" * 60)

# 9. Đóng gói & Lưu trữ Artifacts cho MLOps Pipeline
os.makedirs('data', exist_ok=True)
joblib.dump(poly, 'data/poly.joblib')
joblib.dump(std_scaler, 'data/scaler.joblib')
joblib.dump(regressor, 'data/model.joblib')

print("\nĐã lưu thành công poly.joblib, scaler.joblib và model.joblib vào thư mục data/\n")