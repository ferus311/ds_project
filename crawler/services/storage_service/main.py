import os
import sys
import time
import signal
import logging
import pandas as pd
import io
import math
from datetime import datetime

# Thêm thư mục gốc vào sys.path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from common.queue.kafka_client import KafkaConsumer
from common.utils.logging_utils import setup_logging
from common.storage.hdfs_writer import HDFSWriter
from dotenv import load_dotenv

load_dotenv()

logger = setup_logging()


class StorageService:
    def __init__(self):
        self.running = True
        self.consumer = KafkaConsumer(
            ["property-data"], group_id="storage-service-group"
        )
        self.hdfs_writer = HDFSWriter()
        self.buffer = []

        # Cấu hình từ biến môi trường
        self.buffer_size = int(os.environ.get("BUFFER_SIZE", "50000"))
        self.flush_interval = int(os.environ.get("FLUSH_INTERVAL", "300"))  # 5 phút
        self.max_file_size_mb = int(
            os.environ.get("MAX_FILE_SIZE_MB", "128")
        )  # Kích thước file tối đa (MB)
        self.min_file_size_mb = int(
            os.environ.get("MIN_FILE_SIZE_MB", "1")
        )  # Kích thước file tối thiểu (MB)
        self.optimal_file_size_mb = int(
            os.environ.get("OPTIMAL_FILE_SIZE_MB", "64")
        )  # Kích thước file tối ưu (MB)

        # Số file mà một lần ghi có thể chia nhỏ tối đa
        self.max_split_files = int(os.environ.get("MAX_SPLIT_FILES", "10"))

        self.last_flush_time = time.time()
        # Đếm số lần phân chia file trong một lần ghi
        self.current_split_count = 0
        # Thống kê kích thước file
        self.file_size_stats = {
            "total_files": 0,
            "total_size_mb": 0,
            "avg_size_mb": 0,
            "min_size_mb": float("inf"),
            "max_size_mb": 0,
        }

        # Định dạng lưu trữ: parquet, csv, hoặc json
        self.storage_format = os.environ.get("STORAGE_FORMAT", "parquet").lower()
        # Đường dẫn lưu trữ dữ liệu
        self.output_path = os.environ.get("OUTPUT_PATH", "/data/property_data")
        logger.info(
            f"Using storage format: {self.storage_format}, output path: {self.output_path}, "
            f"optimal file size: {self.optimal_file_size_mb}MB (min: {self.min_file_size_mb}MB, max: {self.max_file_size_mb}MB)"
        )

        # Xử lý tín hiệu để dừng service an toàn
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)

    def stop(self, *args):
        """Xử lý khi nhận tín hiệu dừng"""
        logger.info("Stopping Storage Service...")
        self.running = False
        # Flush buffer trước khi dừng để đảm bảo không mất dữ liệu
        if self.buffer:
            self.flush_buffer(force=True)

        # In thống kê về kích thước file đã ghi
        self._log_file_size_stats()

    def _log_file_size_stats(self):
        """Ghi log thống kê về kích thước file"""
        stats = self.file_size_stats
        if stats["total_files"] > 0:
            logger.info(f"=== File Size Statistics ===")
            logger.info(f"Total files written: {stats['total_files']}")
            logger.info(f"Total data size: {stats['total_size_mb']:.2f} MB")
            logger.info(f"Average file size: {stats['avg_size_mb']:.2f} MB")
            logger.info(f"Minimum file size: {stats['min_size_mb']:.2f} MB")
            logger.info(f"Maximum file size: {stats['max_size_mb']:.2f} MB")

    def _update_file_size_stats(self, size_mb):
        """Cập nhật thống kê kích thước file"""
        stats = self.file_size_stats
        stats["total_files"] += 1
        stats["total_size_mb"] += size_mb
        stats["avg_size_mb"] = stats["total_size_mb"] / stats["total_files"]
        stats["min_size_mb"] = min(stats["min_size_mb"], size_mb)
        stats["max_size_mb"] = max(stats["max_size_mb"], size_mb)

    def estimate_buffer_size_mb(self, sample_records=None):
        """Ước tính kích thước của buffer hiện tại (MB)"""
        if not self.buffer:
            return 0

        if sample_records is None:
            # Lấy mẫu để ước tính kích thước
            sample_size = min(len(self.buffer), 100)
            sample_records = self.buffer[:sample_size]

        try:
            # Chuyển mẫu thành DataFrame
            df_sample = pd.DataFrame(sample_records)

            # Ước tính kích thước dựa trên định dạng lưu trữ
            if self.storage_format == "parquet":
                # Sử dụng bộ nhớ ảo để ước tính kích thước parquet
                parquet_buffer = io.BytesIO()
                df_sample.to_parquet(parquet_buffer, engine="pyarrow")
                sample_size_bytes = parquet_buffer.tell()
            elif self.storage_format == "csv":
                # Ước tính kích thước CSV
                csv_buffer = io.StringIO()
                df_sample.to_csv(csv_buffer, index=False)
                sample_size_bytes = len(csv_buffer.getvalue().encode("utf-8"))
            else:  # json
                # Ước tính kích thước JSON
                json_str = df_sample.to_json(orient="records")
                sample_size_bytes = len(json_str.encode("utf-8"))

            # Ước tính kích thước trung bình mỗi bản ghi
            avg_record_size = sample_size_bytes / len(sample_records)

            # Ước tính tổng kích thước của toàn bộ buffer
            estimated_total_bytes = avg_record_size * len(self.buffer)

            # Chuyển đổi sang MB
            estimated_total_mb = estimated_total_bytes / (1024 * 1024)

            return estimated_total_mb
        except Exception as e:
            logger.warning(f"Error estimating buffer size: {e}")
            # Trả về ước tính thấp nếu có lỗi
            return 0.1 * len(self.buffer) / 1000  # Giả sử mỗi bản ghi khoảng 0.1KB

    def process_message(self, message):
        """Xử lý một message từ Kafka và đưa vào buffer"""
        if not message:
            return False

        try:
            # Thêm message vào buffer
            self.buffer.append(message)

            # Ước tính kích thước hiện tại của buffer
            estimated_size_mb = self.estimate_buffer_size_mb()

            # Kiểm tra các điều kiện để flush:
            # 1. Buffer đã đạt kích thước tối ưu
            # 2. Đã đạt số lượng bản ghi tối đa
            # 3. Đã đến thời gian flush định kỳ và buffer đủ lớn
            should_flush = (
                estimated_size_mb >= self.optimal_file_size_mb
                or len(self.buffer) >= self.buffer_size
                or (
                    time.time() - self.last_flush_time >= self.flush_interval
                    and estimated_size_mb >= self.min_file_size_mb
                )
            )

            if should_flush:
                logger.info(
                    f"Buffer size: {len(self.buffer)} records, estimated {estimated_size_mb:.2f}MB - Flushing to HDFS"
                )
                self.flush_buffer()

            return True

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return False

    def _split_dataframe(self, df, num_parts):
        """Chia DataFrame thành nhiều phần gần bằng nhau"""
        total_rows = len(df)
        rows_per_part = math.ceil(total_rows / num_parts)
        return [
            df.iloc[i : i + rows_per_part] for i in range(0, total_rows, rows_per_part)
        ]

    def flush_buffer(self, force=False):
        """Ghi buffer hiện tại vào HDFS theo định dạng đã cấu hình"""
        if not self.buffer and not force:
            return

        try:
            # Tạo DataFrame từ buffer
            df = pd.DataFrame(self.buffer) if self.buffer else pd.DataFrame()

            if df.empty and not force:
                logger.warning("Empty buffer, skipping flush")
                return

            # Ước tính kích thước của DataFrame
            estimated_size_mb = self.estimate_buffer_size_mb()

            # Nếu kích thước vượt quá ngưỡng tối đa, chia nhỏ file
            if estimated_size_mb > self.max_file_size_mb:
                # Tính số file cần chia dựa trên kích thước ước tính và kích thước tối ưu
                num_splits = min(
                    math.ceil(estimated_size_mb / self.optimal_file_size_mb),
                    self.max_split_files,
                )

                logger.info(
                    f"File size {estimated_size_mb:.2f}MB exceeds maximum {self.max_file_size_mb}MB. "
                    + f"Splitting into {num_splits} parts."
                )

                # Chia DataFrame thành nhiều phần
                df_parts = self._split_dataframe(df, num_splits)

                # Reset biến đếm phân chia file
                self.current_split_count = 0

                # Ghi từng phần vào HDFS
                for part_df in df_parts:
                    self._write_dataframe_to_hdfs(part_df)
            else:
                # Ghi toàn bộ DataFrame vào HDFS
                self._write_dataframe_to_hdfs(df)

            # Xóa buffer sau khi ghi thành công
            self.buffer = []
            self.last_flush_time = time.time()

        except Exception as e:
            logger.error(f"Error flushing buffer to HDFS: {e}")

    def _write_dataframe_to_hdfs(self, df):
        """Ghi một DataFrame vào HDFS và cập nhật thống kê"""
        if df.empty:
            return

        # Ước tính kích thước của DataFrame hiện tại
        current_df_size_mb = self.estimate_buffer_size_mb(df.to_dict("records"))

        # Tạo tên file dựa trên timestamp và split count nếu cần
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        split_suffix = (
            f"_part{self.current_split_count}" if self.current_split_count > 0 else ""
        )
        self.current_split_count += 1

        success = False
        filename = ""

        try:
            # Ghi dữ liệu theo định dạng đã cấu hình
            if self.storage_format == "json":
                filename = f"property_data_{timestamp}{split_suffix}.json"
                logger.info(
                    f"Writing {len(df)} records ({current_df_size_mb:.2f}MB) to HDFS as {filename}"
                )
                success = self.hdfs_writer.write_json(df.to_dict("records"), filename)
            elif self.storage_format == "parquet":
                filename = f"property_data_{timestamp}{split_suffix}.parquet"
                logger.info(
                    f"Writing {len(df)} records ({current_df_size_mb:.2f}MB) to HDFS as {filename}"
                )
                success = self.hdfs_writer.write_parquet(df, filename)
            elif self.storage_format == "csv":
                filename = f"property_data_{timestamp}{split_suffix}.csv"
                logger.info(
                    f"Writing {len(df)} records ({current_df_size_mb:.2f}MB) to HDFS as {filename}"
                )
                success = self.hdfs_writer.write_csv(df, filename)
            else:
                # Định dạng không được hỗ trợ, sử dụng JSON làm mặc định
                logger.warning(
                    f"Unsupported format {self.storage_format}, using JSON instead"
                )
                filename = f"property_data_{timestamp}{split_suffix}.json"
                success = self.hdfs_writer.write_json(df.to_dict("records"), filename)

            if success:
                logger.info(
                    f"Successfully wrote {len(df)} records ({current_df_size_mb:.2f}MB) to HDFS as {filename}"
                )
                # Cập nhật thống kê kích thước file
                self._update_file_size_stats(current_df_size_mb)
            else:
                logger.error(f"Failed to write data to HDFS")

        except Exception as e:
            logger.error(f"Error writing DataFrame to HDFS: {e}")

    def run(self):
        """Chạy service liên tục"""
        logger.info("Storage Service started")

        try:
            while self.running:
                # Tiêu thụ message từ Kafka
                message = self.consumer.consume(timeout=1.0)

                if message:
                    self.process_message(message)
                else:
                    # Nếu không có message mới, kiểm tra xem có cần flush không
                    current_time = time.time()
                    if (
                        current_time - self.last_flush_time >= self.flush_interval
                        and self.buffer
                    ):
                        estimated_size_mb = self.estimate_buffer_size_mb()
                        if estimated_size_mb >= self.min_file_size_mb:
                            logger.info(
                                f"Flush interval reached. Buffer size: {len(self.buffer)} records, "
                                + f"estimated {estimated_size_mb:.2f}MB - Flushing to HDFS"
                            )
                            self.flush_buffer()
                    time.sleep(1)  # Tránh sử dụng CPU quá cao

        except KeyboardInterrupt:
            logger.info("Received interrupt signal. Stopping...")
        except Exception as e:
            logger.error(f"Error in Storage Service: {e}")
        finally:
            # Đảm bảo dữ liệu còn lại được ghi vào HDFS
            if self.buffer:
                self.flush_buffer(force=True)

            # In thống kê kích thước file
            self._log_file_size_stats()

            self.consumer.close()
            logger.info("Storage Service stopped")


def main():
    service = StorageService()
    service.run()


if __name__ == "__main__":
    main()
