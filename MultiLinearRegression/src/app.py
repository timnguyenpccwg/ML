import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("***** MULTI LINEAR REGRESSION *******")
# Tải dữ liệu từ URL[cite: 10]
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
print("start downloading and reading data from provided source")
df = pd.read_csv(url)
print(df.describe())
print(df.info())
print(df.sample(5))


# 2. Tiền xử lý: Loại bỏ các cột phân loại (categorical) và các cột có độ tương quan dư thừa
df = df.drop(['MODELYEAR', 'MAKE', 'MODEL', 'VEHICLECLASS', 'TRANSMISSION', 'FUELTYPE'], axis=1)
df = df.drop(['CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB'], axis=1)

# 3. Trích xuất đặc trưng (X) và biến mục tiêu (y)
# X gồm 2 biến: ENGINESIZE và FUELCONSUMPTION_COMB_MPG
X = df.iloc[:, [0,1]].to_numpy()
y = df.iloc[:, [2]].to_numpy()

# 4. Chuẩn hóa dữ liệu (Standardization)
# Đưa các đặc trưng về cùng thang đo (trung bình = 0, độ lệch chuẩn = 1) để mô hình không bị thiên vị
std_scaler = preprocessing.StandardScaler()
X_std = std_scaler.fit_transform(X)

# 5. Chia tách dữ liệu: 80% train, 20% test[cite: 8]
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.2, random_state=42)

# 6. Huấn luyện mô hình Multiple Linear Regression[cite: 8]
regressor = linear_model.LinearRegression()
regressor.fit(X_train, y_train)

# 7. In ra hệ số và điểm cắt trên không gian đã chuẩn hóa[cite: 8]
coef_ = regressor.coef_
intercept_ = regressor.intercept_
print('Coefficients (Standardized):', coef_)
print('Intercept (Standardized):', intercept_)

# 8. Chuyển đổi hệ số về không gian gốc để dễ diễn giải thực tế[cite: 8]
means_ = std_scaler.mean_
std_devs_ = np.sqrt(std_scaler.var_)
coef_original = coef_ / std_devs_
intercept_original = intercept_ - np.sum((means_ * coef_) / std_devs_)

print('\nCoefficients (Original Space):', coef_original)
print('Intercept (Original Space):', intercept_original)

# 9. Đánh giá mô hình trên tập Test
y_pred = regressor.predict(X_test)
print("\n--- Model Evaluation ---")
print("Mean absolute error: %.2f" % mean_absolute_error(y_test, y_pred))
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print("R2-score: %.2f" % r2_score(y_test, y_pred))