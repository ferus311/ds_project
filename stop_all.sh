#!/bin/bash
echo "===== DỪNG HỆ THỐNG THU THẬP DỮ LIỆU BẤT ĐỘNG SẢN ====="

cd ./docker

# Dừng từng dịch vụ theo thứ tự ngược lại
echo "Dừng các crawler service..."
docker compose -f yml/crawler.yml down

echo "Dừng Spark..."
docker compose -f yml/spark.yml down

echo "Dừng Airflow..."
docker compose -f yml/airflow.yml down

echo "Dừng Kafka..."
docker compose -f yml/kafka.yml down

echo "Dừng HDFS..."
docker compose -f yml/hdfs.yml down

echo "===== ĐÃ DỪNG TOÀN BỘ HỆ THỐNG ====="
