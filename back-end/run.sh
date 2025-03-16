#!/bin/bash

# Màu sắc cho output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Bắt đầu quá trình deploy...${NC}"

# Kiểm tra Docker đã được cài đặt chưa
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker chưa được cài đặt. Vui lòng cài đặt Docker trước.${NC}"
    exit 1
fi

# Kiểm tra Docker Compose đã được cài đặt chưa
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Docker Compose chưa được cài đặt. Vui lòng cài đặt Docker Compose trước.${NC}"
    exit 1
fi

# Kiểm tra file .env tồn tại
if [ ! -f .env ]; then
    echo -e "${RED}File .env không tồn tại. Vui lòng tạo file .env với các biến môi trường cần thiết.${NC}"
    exit 1
fi

# Dừng các container đang chạy (nếu có)
echo -e "${GREEN}Dừng các container đang chạy...${NC}"
docker-compose down

# Xóa các image cũ (nếu có)
echo -e "${GREEN}Xóa các image cũ...${NC}"
docker-compose rm -f

# Build lại image
echo -e "${GREEN}Build lại Docker image...${NC}"
docker-compose build --no-cache

# Chạy container
echo -e "${GREEN}Khởi động container...${NC}"
docker-compose up -d

# Kiểm tra trạng thái container
echo -e "${GREEN}Kiểm tra trạng thái container...${NC}"
docker-compose ps

# Kiểm tra logs
echo -e "${GREEN}Xem logs của container...${NC}"
docker-compose logs -f 