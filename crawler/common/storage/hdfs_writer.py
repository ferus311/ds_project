import os
import json
import logging
import tempfile
import numpy as np
import pandas as pd
from datetime import datetime
from pyhdfs import HdfsClient
import socket
import time

logger = logging.getLogger(__name__)


class HDFSWriter:
    def __init__(self):
        # WebHDFS thường chạy trên cổng 50070 (Hadoop 2.x) hoặc 9870 (Hadoop 3.x)
        # Không sử dụng cổng 9000 vì đó là cổng IPC của Hadoop, không phải WebHDFS
        self.hdfs_hosts = os.environ.get("HDFS_HOSTS", "namenode:9870").split(",")
        self.hdfs_user = os.environ.get("HDFS_USER", "hadoop")
        self.base_path = os.environ.get("HDFS_BASE_PATH", "/real_estate_data")

        # Số lần thử kết nối lại
        self.max_retries = int(os.environ.get("HDFS_MAX_RETRIES", "3"))
        # Thời gian chờ giữa các lần thử kết nối (giây)
        self.retry_interval = int(os.environ.get("HDFS_RETRY_INTERVAL", "5"))

        # Cấu hình kích thước file tối ưu (byte) - mặc định 128MB
        self.target_file_size = int(
            os.environ.get("HDFS_TARGET_FILE_SIZE", 128 * 1024 * 1024)
        )

        # Định dạng file mặc định (parquet hoặc csv)
        self.default_format = os.environ.get("HDFS_DEFAULT_FORMAT", "parquet")

        # Cờ để kiểm tra nếu WebHDFS hoạt động (tránh lặp lại thông báo lỗi)
        self.webhdfs_working = False

        # Kết nối tới HDFS
        self._connect_to_hdfs()

    def _connect_to_hdfs(self):
        """Thiết lập kết nối đến HDFS với cơ chế retry và phục hồi lỗi"""
        retry_count = 0
        connected = False

        while retry_count < self.max_retries and not connected:
            try:
                logger.info(
                    f"Connecting to HDFS at {self.hdfs_hosts} (Attempt {retry_count + 1}/{self.max_retries})"
                )

                # Thử tạo client mới
                self.client = HdfsClient(
                    hosts=self.hdfs_hosts, user_name=self.hdfs_user
                )

                # Kiểm tra kết nối bằng cách thực hiện một thao tác đơn giản
                self.client.get_file_status("/")

                # Nếu đến được đây, kết nối thành công
                logger.info(f"Successfully connected to HDFS at {self.hdfs_hosts}")

                # Tạo thư mục cơ sở nếu chưa tồn tại
                if not self.client.exists(self.base_path):
                    self.client.mkdirs(self.base_path)

                # Đánh dấu WebHDFS đang hoạt động
                self.webhdfs_working = True
                connected = True

            except Exception as e:
                retry_count += 1
                error_msg = str(e)

                # Kiểm tra lỗi cụ thể để đưa ra thông báo hữu ích
                if (
                    "Expected JSON" in error_msg
                    and "It looks like you are making an HTTP request to a Hadoop IPC port"
                    in error_msg
                ):
                    logger.error(
                        f"WebHDFS connection error: You're using an IPC port instead of WebHDFS port. "
                        f"WebHDFS typically runs on port 50070 (Hadoop 2.x) or 9870 (Hadoop 3.x), not 9000."
                    )
                elif "Connection refused" in error_msg:
                    logger.error(
                        f"HDFS connection refused. The HDFS service might be down or the port is incorrect."
                    )
                elif "timed out" in error_msg:
                    logger.error(
                        f"HDFS connection timed out. Check if the HDFS service is running or network connectivity."
                    )
                else:
                    logger.error(f"Error connecting to HDFS: {e}")

                self.client = None

                if retry_count < self.max_retries:
                    logger.info(f"Retrying in {self.retry_interval} seconds...")
                    time.sleep(self.retry_interval)
                else:
                    logger.error(
                        f"Failed to connect to HDFS after {self.max_retries} attempts. "
                        f"Writing to HDFS will be disabled."
                    )

    def _ensure_hdfs_connection(self):
        """Kiểm tra và thử kết nối lại HDFS nếu cần"""
        if self.client is None and not self.webhdfs_working:
            self._connect_to_hdfs()
        return self.client is not None

    def _get_path_for_date(self):
        """Tạo đường dẫn trên HDFS theo cấu trúc phân cấp theo ngày"""
        today = datetime.now().strftime("%Y/%m/%d")
        path = f"{self.base_path}/{today}"

        # Tạo thư mục nếu chưa tồn tại
        if not self.client.exists(path):
            self.client.mkdirs(path)

        return path

    def _estimate_file_size(self, df):
        """Ước tính kích thước DataFrame để quyết định có nên chia file hay không"""
        # Phương pháp ước lượng đơn giản dựa trên memory usage
        memory_usage = df.memory_usage(deep=True).sum()
        logger.info(f"Estimated DataFrame size: {memory_usage} bytes")
        return memory_usage

    def _split_dataframe(self, df, max_size):
        """Chia DataFrame thành các phần nhỏ hơn dựa trên kích thước ước tính"""
        estimated_size = self._estimate_file_size(df)
        if estimated_size <= max_size:
            return [df]

        # Ước tính số file cần chia
        num_chunks = max(2, int(estimated_size / max_size) + 1)
        logger.info(f"Splitting DataFrame into {num_chunks} chunks")

        # Chia DataFrame thành các phần bằng nhau
        return [chunk for chunk in np.array_split(df, num_chunks)]

    def write_json(self, data, filename):
        """Ghi dữ liệu JSON vào HDFS"""
        if not self._ensure_hdfs_connection():
            logger.error("HDFS client not initialized or WebHDFS not available")
            return False

        try:
            # Tạo tệp tin tạm thời cục bộ
            with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
                json.dump(data, temp_file, ensure_ascii=False, indent=2)
                temp_file_path = temp_file.name

            # Đường dẫn đích trên HDFS
            hdfs_path = f"{self._get_path_for_date()}/{filename}"

            # Tải tệp tin lên HDFS
            self.client.copy_from_local(temp_file_path, hdfs_path)

            # Xóa tệp tin tạm thời
            os.unlink(temp_file_path)

            logger.info(f"Successfully wrote data to HDFS at {hdfs_path}")
            return True

        except Exception as e:
            logger.error(f"Error writing to HDFS: {e}")
            return False

    def write_csv(self, df, filename):
        """Ghi DataFrame vào HDFS dưới dạng CSV"""
        if not self._ensure_hdfs_connection():
            logger.error("HDFS client not initialized or WebHDFS not available")
            return False

        try:
            # Kiểm tra kích thước và chia DataFrame nếu cần
            chunks = self._split_dataframe(df, self.target_file_size)

            successful_writes = 0

            for i, chunk in enumerate(chunks):
                # Tạo tên file phù hợp nếu có nhiều phần
                chunk_filename = (
                    filename
                    if len(chunks) == 1
                    else f"{os.path.splitext(filename)[0]}_part{i}{os.path.splitext(filename)[1]}"
                )

                # Tạo tệp tin tạm thời cục bộ
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    chunk.to_csv(temp_file.name, index=False)
                    temp_file_path = temp_file.name

                # Đường dẫn đích trên HDFS
                hdfs_path = f"{self._get_path_for_date()}/{chunk_filename}"

                # Tải tệp tin lên HDFS
                self.client.copy_from_local(temp_file_path, hdfs_path)

                # Xóa tệp tin tạm thời
                os.unlink(temp_file_path)

                successful_writes += 1
                logger.info(
                    f"Successfully wrote CSV chunk {i+1}/{len(chunks)} to HDFS at {hdfs_path}"
                )

            return successful_writes == len(chunks)

        except Exception as e:
            logger.error(f"Error writing CSV to HDFS: {e}")
            return False

    def write_parquet(self, df, filename):
        """Ghi DataFrame vào HDFS dưới dạng Parquet (hiệu quả hơn cho Big Data)"""
        if not self._ensure_hdfs_connection():
            logger.error("HDFS client not initialized or WebHDFS not available")
            return False

        try:
            # Đảm bảo filename có đuôi .parquet
            if not filename.endswith(".parquet"):
                filename = f"{os.path.splitext(filename)[0]}.parquet"

            # Kiểm tra kích thước và chia DataFrame nếu cần
            chunks = self._split_dataframe(df, self.target_file_size)

            successful_writes = 0

            for i, chunk in enumerate(chunks):
                # Tạo tên file phù hợp nếu có nhiều phần
                chunk_filename = (
                    filename
                    if len(chunks) == 1
                    else f"{os.path.splitext(filename)[0]}_part{i}.parquet"
                )

                # Tạo tệp tin tạm thời cục bộ
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    # Ghi DataFrame ra file Parquet với nén snappy (tối ưu tốc độ và kích thước)
                    chunk.to_parquet(
                        temp_file.name,
                        engine="pyarrow",
                        compression="snappy",
                        index=False,
                    )
                    temp_file_path = temp_file.name

                # Đường dẫn đích trên HDFS
                hdfs_path = f"{self._get_path_for_date()}/{chunk_filename}"

                # Tải tệp tin lên HDFS
                self.client.copy_from_local(temp_file_path, hdfs_path)

                # Xóa tệp tin tạm thời
                os.unlink(temp_file_path)

                successful_writes += 1
                logger.info(
                    f"Successfully wrote Parquet chunk {i+1}/{len(chunks)} to HDFS at {hdfs_path}"
                )

            return successful_writes == len(chunks)

        except Exception as e:
            logger.error(f"Error writing Parquet to HDFS: {e}")
            return False

    def write(self, df, filename, format=None):
        """
        Phương thức tiện ích để ghi dữ liệu theo định dạng mặc định hoặc được chỉ định
        """
        if format is None:
            format = self.default_format

        if format.lower() == "parquet":
            return self.write_parquet(df, filename)
        elif format.lower() == "csv":
            return self.write_csv(df, filename)
        else:
            logger.error(
                f"Unsupported file format: {format}. Using default format {self.default_format}"
            )
            return self.write(df, filename, self.default_format)
