import joblib
import pandas as pd

# 1. Đánh thức mô hình và bộ chuẩn hóa từ thư mục data
scaler = joblib.load('data/scaler.joblib')
model = joblib.load('data/model.joblib')

# 2. Loại bỏ mảng ẩn danh: Dùng Dictionary và Pandas để gán đích danh dữ liệu
thong_so_xe = {
    'ENGINESIZE': [2.4],
    'FUELCONSUMPTION_COMB_MPG': [25]
}
xe_moi_df = pd.DataFrame(thong_so_xe)

# Đảm bảo dữ liệu luôn được nạp đúng thứ tự thiết kế, bất chấp vị trí khai báo ở trên
features = ['ENGINESIZE', 'FUELCONSUMPTION_COMB_MPG']
xe_moi_array = xe_moi_df[features].to_numpy()

# 3. Chuẩn hóa dữ liệu thô và dự đoán
xe_moi_std = scaler.transform(xe_moi_array)
co2_du_doan = model.predict(xe_moi_std)

# 4. Loại bỏ magic number [0][0] bằng .item()
ket_qua = co2_du_doan.item()

print(f"Lượng khí thải dự đoán: {ket_qua:.2f} g/km")