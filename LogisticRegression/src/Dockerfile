# Sử dụng Python image chính thức
FROM python:3.12-slim

# Thiết lập thư mục làm việc trong container
WORKDIR /app

# Copy requirements và cài đặt thư viện
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy mã nguồn vào container
COPY app.py .

# Lệnh chạy script khi container khởi động
CMD ["python", "app.py"]