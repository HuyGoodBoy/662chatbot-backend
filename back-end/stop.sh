#!/bin/bash

# Màu sắc cho output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Bắt đầu dừng container...${NC}"

# Kiểm tra xem có container đang chạy không
if ! docker-compose ps | grep -q "Up"; then
    echo -e "${RED}Không có container nào đang chạy.${NC}"
    exit 0
fi

# Dừng container
echo -e "${GREEN}Dừng các container...${NC}"
docker-compose down

# Xóa các container đã dừng
echo -e "${GREEN}Xóa các container đã dừng...${NC}"
docker-compose rm -f

echo -e "${GREEN}Đã dừng và xóa container thành công!${NC}" 