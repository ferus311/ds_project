#!/bin/bash

# Script khởi động hệ thống crawler bất động sản
# Khởi động các dịch vụ theo thứ tự phù hợp
set -e
# cd ./docker
# Màu sắc cho output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Thư mục gốc của dự án
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Đảm bảo biến môi trường AIRFLOW_UID được thiết lập
export AIRFLOW_UID=${AIRFLOW_UID:-$(id -u)}
echo -e "${BLUE}[INFO]${NC} Sử dụng AIRFLOW_UID=${AIRFLOW_UID}"

# Hàm kiểm tra lỗi
check_error() {
    if [ $? -ne 0 ]; then
        echo -e "${RED}[ERROR]${NC} $1"
        exit 1
    fi
}

# Hàm kiểm tra dịch vụ đã sẵn sàng chưa
wait_for_service() {
    local service=$1
    local port=$2
    local host=${3:-localhost}
    local timeout=${4:-60}

    echo -e "${YELLOW}[WAIT]${NC} Đang chờ $service khởi động tại $host:$port..."

    local start_time=$(date +%s)
    local end_time=$((start_time + timeout))

    while [ $(date +%s) -lt $end_time ]; do
        if nc -z $host $port > /dev/null 2>&1; then
            echo -e "${GREEN}[OK]${NC} $service đã sẵn sàng!"
            return 0
        fi
        sleep 2
    done

    echo -e "${RED}[TIMEOUT]${NC} $service không khởi động sau $timeout giây."
    return 1
}

# Tạo thư mục volumes nếu chưa tồn tại
mkdir -p ${PROJECT_DIR}/docker/volumes/hdfs/namenode
mkdir -p ${PROJECT_DIR}/docker/volumes/hdfs/datanode1
# mkdir -p ${PROJECT_DIR}/docker/volumes/hdfs/datanode2
# mkdir -p ${PROJECT_DIR}/docker/volumes/hdfs/datanode3

# Kiểm tra xem các networks đã tồn tại chưa
echo -e "${BLUE}[INFO]${NC} Kiểm tra và tạo Docker networks..."
if ! docker network ls | grep -q hdfs_network; then
    echo -e "${BLUE}[INFO]${NC} Tạo network hdfs_network..."
    docker network create hdfs_network
    check_error "Không thể tạo network hdfs_network"
fi

if ! docker network ls | grep -q kafka_network; then
    echo -e "${BLUE}[INFO]${NC} Tạo network kafka_network..."
    docker network create kafka_network
    check_error "Không thể tạo network kafka_network"
fi

# Khởi động HDFS cluster
echo -e "${BLUE}[INFO]${NC} Khởi động HDFS cluster..."
docker compose -f ${PROJECT_DIR}/docker/yml/hdfs.yml up -d namenode
check_error "Không thể khởi động namenode"

# Đợi namenode khởi động
wait_for_service "namenode" "9870" "localhost" 120

# Khởi động các datanode
echo -e "${BLUE}[INFO]${NC} Khởi động các datanode..."
docker compose -f ${PROJECT_DIR}/docker/yml/hdfs.yml up -d datanode1
check_error "Không thể khởi động datanode"

# Khởi động Kafka
echo -e "${BLUE}[INFO]${NC} Khởi động Zookeeper và Kafka..."
docker compose -f ${PROJECT_DIR}/docker/yml/kafka.yml up -d
check_error "Không thể khởi động Kafka"

# Đợi Kafka khởi động
wait_for_service "Kafka" "9092" "localhost" 120

# Khởi động Airflow
echo -e "${BLUE}[INFO]${NC} Khởi động Airflow..."
docker compose -f ${PROJECT_DIR}/docker/yml/airflow.yml up -d
check_error "Không thể khởi động Airflow"

# Đợi Airflow webserver khởi động
wait_for_service "Airflow webserver" "8080" "localhost" 180

# Khởi động crawler shell
echo -e "${BLUE}[INFO]${NC} Khởi động crawler shell..."
docker compose -f ${PROJECT_DIR}/docker/yml/crawler.yml up -d crawler-shell
check_error "Không thể khởi động crawler shell"

echo -e "${GREEN}[SUCCESS]${NC} Tất cả các dịch vụ đã được khởi động thành công!"
echo -e "${GREEN}[INFO]${NC} Airflow UI: http://localhost:8080 (admin/admin)"
echo -e "${GREEN}[INFO]${NC} HDFS UI: http://localhost:9870"
echo -e "${GREEN}[INFO]${NC} Kafka UI: http://localhost:8282"
echo -e "${GREEN}[INFO]${NC} Để truy cập crawler shell: docker exec -it crawler-shell bash"
