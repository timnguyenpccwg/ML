import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    classification_report
)

# --- 1. CẤU HÌNH HẰNG SỐ ---
DATA_PATH = 'data/ChurnData.csv'  # Đường dẫn file dữ liệu địa phương hoặc URL
URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv"
FEATURES = ['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']
TARGET = 'churn'
TEST_SIZE = 0.2
RANDOM_STATE = 42
ARTIFACT_DIR = 'data/churnprediction'
MODEL_OUTPUT = os.path.join(ARTIFACT_DIR, 'churn_pipeline.joblib')

def load_data():
    """Tải dữ liệu từ file local hoặc URL nếu chưa có sẵn"""
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    else:
        print(f"Đang tải dữ liệu từ URL...")
        df = pd.read_csv(URL)
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
    return df

def main():
    # --- 2. NẠP VÀ TIỀN XỬ LÝ SƠ BỘ ---
    df = load_data()
    X = df[FEATURES]
    y = df[TARGET].astype(int)

    # --- 3. CHIA TÁCH DỮ LIỆU (CHỐNG DATA LEAKAGE + STRATIFY) ---
    # Sử dụng stratify=y để giữ nguyên tỷ lệ nhãn (0/1) ở cả 2 tập Train và Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE, 
        stratify=y
    )

    # --- 4. XÂY DỰNG PIPELINE KHẮC PHỤC LỖI RÒ RỈ DỮ LIỆU ---
    # Pipeline đảm bảo StandardScaler chỉ học (fit) từ X_train
    churn_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=RANDOM_STATE, solver='liblinear'))
    ])

    # --- 5. HUẤN LUYỆN ---
    print("Bắt đầu huấn luyện Churn Classification Pipeline...")
    churn_pipeline.fit(X_train, y_train)

    # --- 6. ĐÁNH GIÁ MÔ HÌNH TOÀN DIỆN ---
    y_pred = churn_pipeline.predict(X_test)
    y_prob = churn_pipeline.predict_proba(X_test)[:, 1]  # Xác suất rời bỏ dịch vụ (Class 1)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    loss = log_loss(y_test, y_prob)

    print("=" * 60)
    print("BÁO CÁO ĐÁNH GIÁ MÔ HÌNH CHURN CLASSIFICATION")
    print("=" * 60)
    print(f"Accuracy         : {acc * 100:.2f}%")
    print(f"Precision (1)    : {prec * 100:.2f}%")
    print(f"Recall (1)       : {rec * 100:.2f}%")
    print(f"F1-Score (1)     : {f1 * 100:.2f}%")
    print(f"ROC-AUC Score    : {roc_auc * 100:.2f}%")
    print(f"Log Loss         : {loss:.4f}")
    print("=" * 60)
    print("\nChi tiết Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=['Stay (0)', 'Churn (1)']))

    # --- 7. ĐÓNG GÓI ARTIFACT NGUYÊN KHỐI ---
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    joblib.dump(churn_pipeline, MODEL_OUTPUT)
    print(f"Đã xuất thành công pipeline hoàn chỉnh tại: {MODEL_OUTPUT}")

if __name__ == "__main__":
    main()