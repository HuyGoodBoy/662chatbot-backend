# Sử dụng Python 3.9 làm base image
FROM python:3.10-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# Cài đặt các package system cần thiết
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements trước để tận dụng cache của Docker
COPY requirements.txt .

# Cài đặt dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose port
EXPOSE 8000

# Command để chạy ứng dụng
CMD ["uvicorn", "lol:app_api", "--host", "0.0.0.0", "--port", "8000"]
