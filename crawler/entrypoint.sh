#!/bin/bash
set -e

# Create necessary directories
echo "Creating necessary directories and setting permissions..."
mkdir -p /app/logs
chmod -R 777 /app/logs

# Wait for Kafka to be ready
echo "Waiting for Kafka..."
until nc -z ${KAFKA_HOST:-kafka} ${KAFKA_PORT:-29092}; do
  echo "Kafka is unavailable - sleeping"
  sleep 2
done
echo "Kafka is up"

# Wait for HDFS to be ready
echo "Waiting for HDFS..."
until nc -z ${HDFS_HOST:-namenode} ${HDFS_PORT:-9000}; do
  echo "HDFS is unavailable - sleeping"
  sleep 2
done
echo "HDFS is up"

# Start the specified service or default to list crawler
SERVICE=${1:-list_crawler}
COMMAND=${2:-run}
ADDITIONAL_ARGS=${@:3}

echo "Starting $SERVICE service with command: $COMMAND"
case $SERVICE in
  list_crawler)
    exec python -m services.list_crawler.main $ADDITIONAL_ARGS
    ;;
  detail_crawler)
    exec python -m services.detail_crawler.main $ADDITIONAL_ARGS
    ;;
  storage_service)
    exec python -m services.storage_service.main $ADDITIONAL_ARGS
    ;;
  retry_service)
    exec python -m services.retry_service.main $ADDITIONAL_ARGS
    ;;
  *)
    echo "Unknown service: $SERVICE"
    echo "Available services: list_crawler, detail_crawler, storage_service, retry_service"
    exit 1
    ;;
esac
