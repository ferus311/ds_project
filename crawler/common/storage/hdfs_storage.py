import os
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import json
import logging
from hdfs import InsecureClient
from datetime import datetime
from typing import Dict, List, Optional, Union, Any

logger = logging.getLogger(__name__)

class HDFSStorage:
    def __init__(self, namenode=None, user=None, base_path=None):
        """Khởi tạo kết nối đến HDFS"""
        self.namenode = namenode or os.environ.get("HDFS_NAMENODE", "namenode:9870")
        self.user = user or os.environ.get("HDFS_USER", "airflow")
        self.base_path = base_path or os.environ.get("HDFS_BASE_PATH", "/data/realestate")
        self.client = InsecureClient(f'http://{self.namenode}', user=self.user)
        logger.info(f"Initialized HDFS connection to {self.namenode}")

    def upload_file(self, local_path: str, hdfs_path: Optional[str] = None) -> str:
        """Upload file từ local lên HDFS"""
        if hdfs_path is None:
            filename = os.path.basename(local_path)
            hdfs_path = f"{self.base_path}/{filename}"

        # Đảm bảo thư mục cha tồn tại
        parent_dir = os.path.dirname(hdfs_path)
        self.makedirs(parent_dir)

        # Upload file
        self.client.upload(hdfs_path, local_path, overwrite=True)
        logger.info(f"Uploaded {local_path} to HDFS:{hdfs_path}")
        return hdfs_path

    def read_file(self, hdfs_path: str) -> str:
        """Đọc nội dung file từ HDFS"""
        if not self.check_exists(hdfs_path):
            raise FileNotFoundError(f"File not found in HDFS: {hdfs_path}")

        with self.client.read(hdfs_path) as reader:
            content = reader.read().decode('utf-8')
        return content

    def write_file(self, hdfs_path: str, content: str) -> str:
        """Ghi nội dung vào file trên HDFS"""
        # Đảm bảo thư mục cha tồn tại
        parent_dir = os.path.dirname(hdfs_path)
        self.makedirs(parent_dir)

        # Ghi file
        with self.client.write(hdfs_path, overwrite=True) as writer:
            writer.write(content.encode('utf-8'))

        logger.info(f"Wrote content to HDFS:{hdfs_path}")
        return hdfs_path

    def append_file(self, hdfs_path: str, content: str) -> bool:
        """Thêm nội dung vào file hiện có"""
        try:
            # Kiểm tra file tồn tại
            if not self.check_exists(hdfs_path):
                return self.write_file(hdfs_path, content)

            # Đọc nội dung hiện tại
            current_content = self.read_file(hdfs_path)

            # Ghi lại với nội dung mới
            new_content = current_content + content
            self.write_file(hdfs_path, new_content)

            return hdfs_path
        except Exception as e:
            logger.error(f"Error appending to file {hdfs_path}: {e}")
            return False

    def check_exists(self, path: str) -> bool:
        """Kiểm tra xem path có tồn tại trên HDFS không"""
        return self.client.status(path, strict=False) is not None

    def makedirs(self, path: str) -> bool:
        """Tạo thư mục và các thư mục cha nếu chưa tồn tại"""
        try:
            self.client.makedirs(path)
            return True
        except Exception as e:
            logger.error(f"Error creating directories {path}: {e}")
            return False

    def save_dataframe_as_parquet(self, df: pd.DataFrame, hdfs_path: Optional[str] = None) -> str:
        """Lưu DataFrame dưới dạng Parquet trên HDFS"""
        if hdfs_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            hdfs_path = f"{self.base_path}/properties_{timestamp}.parquet"

        # Tạo file tạm local
        local_path = f"/tmp/properties_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"

        try:
            # Chuyển đổi sang PyArrow Table và lưu local
            table = pa.Table.from_pandas(df)
            pq.write_table(table, local_path)

            # Upload lên HDFS
            result_path = self.upload_file(local_path, hdfs_path)

            # Xóa file tạm
            if os.path.exists(local_path):
                os.remove(local_path)

            return result_path
        except Exception as e:
            logger.error(f"Error saving DataFrame to HDFS: {e}")
            if os.path.exists(local_path):
                os.remove(local_path)
            raise
