import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Tải dữ liệu từ URL[cite: 10]
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
print("start downloading and reading data from provided source")
df = pd.read_csv(url)
print(df.describe())
print(df.info())
print(df.sample(5))


# Chọn đặc trưng ENGINESIZE để dự đoán CO2EMISSIONS[cite: 10]
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
X = cdf.ENGINESIZE.to_numpy()
y = cdf.CO2EMISSIONS.to_numpy()

# Chia dữ liệu: 80% train, 20% test[cite: 10]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Xây dựng và huấn luyện mô hình Hồi quy tuyến tính[cite: 10]
regressor = linear_model.LinearRegression()
regressor.fit(X_train.reshape(-1, 1), y_train)

print('Coefficients: ', regressor.coef_[0])
print('Intercept: ', regressor.intercept_)

# Dự đoán và Đánh giá[cite: 10]
y_pred = regressor.predict(X_test.reshape(-1, 1))
print("Mean absolute error: %.2f" % mean_absolute_error(y_test, y_pred))
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print("R2-score: %.2f" % r2_score(y_test, y_pred))