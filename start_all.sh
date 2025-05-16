#!/bin/bash

# Đặt biến để dễ debug
set -e  # Thoát script nếu có lỗi
START_TIME=$(date +%s)

echo "===== BẮT ĐẦU KHỞI ĐỘNG HỆ THỐNG THU THẬP DỮ LIỆU BẤT ĐỘNG SẢN ====="
echo "Thời gian bắt đầu: $(date)"

# Danh sách các thư mục cần thiết
dirs=(
  "docker/volumes/kafka"
  "docker/volumes/zookeeper"
  "docker/volumes/hdfs/namenode"
  "docker/volumes/hdfs/datanode1"
  "docker/volumes/hdfs/datanode2"
  "docker/volumes/hdfs/datanode3"
  "data_processing/airflow/dags"
  "data_processing/airflow/logs"
  "data_processing/airflow/plugins"
  "data_processing/notebooks"
  "data_processing/spark/jobs"
)

# Kiểm tra xem có thiếu thư mục nào không
need_init=false
for dir in "${dirs[@]}"; do
  if [ ! -d "$dir" ]; then
    echo "⚠️  Thiếu thư mục: $dir"
    need_init=true
    break
  fi
done

# Nếu thiếu, gọi script init
if $need_init; then
  echo "🚧 Đang chạy init_volumes.sh để tạo thư mục..."
  ./scripts/init_volumes.sh
else
  echo "✅ Tất cả thư mục đã tồn tại. Bỏ qua bước init."
fi


# Kiểm tra file hadoop.env
HADOOP_ENV_FILE="docker/hadoop.env"

if [ ! -f "$HADOOP_ENV_FILE" ]; then
    echo "CẢNH BÁO: Không tìm thấy file hadoop.env tại $HADOOP_ENV_FILE."
    exit 1
fi

# Di chuyển đến thư mục docker
cd docker

# Khởi chạy hệ thống từng phần
echo "===== KHỞI ĐỘNG HỆ THỐNG ====="

# 1. Khởi động HDFS
echo "===== KHỞI ĐỘNG HDFS ====="
docker compose -f yml/hdfs.yml up -d
echo "Đợi HDFS khởi động hoàn tất..."
sleep 5

# Kiểm tra HDFS đã khởi động thành công
if docker ps | grep namenode > /dev/null; then
    echo "✅ HDFS namenode đã khởi động thành công."
else
    echo "❌ HDFS namenode khởi động thất bại. Kiểm tra logs: docker logs namenode"
    exit 1
fi

# 2. Khởi động Kafka và ZooKeeper
echo "===== KHỞI ĐỘNG KAFKA ====="
docker compose -f yml/kafka.yml up -d
echo "Đợi Kafka khởi động hoàn tất..."
sleep 5

# Kiểm tra Kafka đã khởi động thành công
if docker ps | grep kafka > /dev/null; then
    echo "✅ Kafka đã khởi động thành công."
else
    echo "❌ Kafka khởi động thất bại. Kiểm tra logs: docker logs kafka"
    exit 1
fi

# 3. Khởi động Airflow
echo "===== KHỞI ĐỘNG AIRFLOW ====="
# Thiết lập Airflow
docker compose -f yml/airflow.yml up -d
sleep 5
# Kiểm tra Airflow đã khởi động thành công
if docker ps | grep airflow-web > /dev/null; then
    echo "✅ Airflow webserver đã khởi động thành công."
else
    echo "❌ Airflow webserver khởi động thất bại. Kiểm tra logs: docker logs airflow-web"
fi

# 4. Khởi động Spark
echo "===== KHỞI ĐỘNG SPARK ====="
docker compose -f yml/spark.yml up -d
echo "Đợi Spark khởi động hoàn tất..."
sleep 5

# Kiểm tra Spark đã khởi động thành công
if docker ps | grep spark-master > /dev/null; then
    echo "✅ Spark master đã khởi động thành công."
else
    echo "❌ Spark master khởi động thất bại. Kiểm tra logs: docker logs spark-master"
fi

# 5. Khởi động các crawler service
echo "===== KHỞI ĐỘNG CRAWLER SERVICES ====="
docker compose -f yml/crawler.yml up -d
echo "Đợi các crawler service khởi động hoàn tất..."
sleep 5

# Kiểm tra các dịch vụ chính đã hoạt động chưa
echo "===== KIỂM TRA TRẠNG THÁI DỊCH VỤ ====="

# Tính thời gian khởi động
END_TIME=$(date +%s)
RUNTIME=$((END_TIME-START_TIME))
MINUTES=$((RUNTIME / 60))
SECONDS=$((RUNTIME % 60))

echo "===== HOÀN THÀNH KHỞI ĐỘNG ====="
echo "Tất cả các dịch vụ đã được khởi động trong $MINUTES phút $SECONDS giây!"
echo
echo "TRUY CẬP CÁC GIAO DIỆN:"
echo "- HDFS UI: http://localhost:9870"
echo "- Airflow UI: http://localhost:8080 (user/pass: admin/admin)"
echo "- Spark UI: http://localhost:8181"
echo "- Jupyter: http://localhost:8888"
echo
echo "KIỂM TRA TRẠNG THÁI:"
echo "- Xem danh sách container: docker ps"
echo "- Xem logs HDFS: docker logs namenode"
echo "- Xem logs Kafka: docker logs kafka"
echo "- Xem logs Airflow: docker logs airflow-web"
echo "- Xem logs crawler: docker logs list-crawler"
echo
echo "DỪNG HỆ THỐNG:"
echo "- Sử dụng: ./scripts/stop_all.sh"
