.PHONY: build run clean

# Build image bằng podman-compose
build:
	podman-compose build

# Khởi chạy container (tự động build nếu image chưa tồn tại)
run:
	podman-compose up

# Dọn dẹp container và dọn dẹp image rác
clean:
	podman-compose down
	podman rmi ml-simple-regression:latest