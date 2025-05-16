#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from pyspark.sql import SparkSession, DataFrame
import argparse
import os
from typing import Dict, Any, Optional, List


class BaseDataProcessor(ABC):
    """
    Lớp cơ sở cho tất cả các bộ xử lý dữ liệu
    """

    def __init__(self, app_name: str = "Data Processor"):
        self.app_name = app_name
        self._spark = None

    @property
    def spark(self) -> SparkSession:
        """
        Khởi tạo và trả về SparkSession
        """
        if self._spark is None:
            self._spark = (
                SparkSession.builder.appName(self.app_name)
                .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
                .getOrCreate()
            )
        return self._spark

    @abstractmethod
    def clean_data(self, df: DataFrame) -> DataFrame:
        """
        Làm sạch dữ liệu

        Args:
            df: DataFrame cần làm sạch

        Returns:
            DataFrame: DataFrame đã được làm sạch
        """
        pass

    @abstractmethod
    def transform_data(self, df: DataFrame) -> DataFrame:
        """
        Biến đổi dữ liệu

        Args:
            df: DataFrame cần biến đổi

        Returns:
            DataFrame: DataFrame đã được biến đổi
        """
        pass

    @abstractmethod
    def feature_engineering(self, df: DataFrame) -> DataFrame:
        """
        Tạo các đặc trưng mới

        Args:
            df: DataFrame cần tạo đặc trưng

        Returns:
            DataFrame: DataFrame với các đặc trưng mới
        """
        pass

    def load_data(self, input_path: str) -> DataFrame:
        """
        Đọc dữ liệu từ đường dẫn

        Args:
            input_path: Đường dẫn đến dữ liệu

        Returns:
            DataFrame: DataFrame đã đọc
        """
        if not input_path.startswith("hdfs://"):
            input_path = os.path.abspath(input_path)

        # Tự động phát hiện định dạng dựa vào phần mở rộng
        if input_path.endswith(".parquet"):
            return self.spark.read.parquet(input_path)
        elif input_path.endswith(".csv"):
            return self.spark.read.option("header", "true").csv(input_path)
        elif input_path.endswith(".json"):
            return self.spark.read.json(input_path)
        else:
            # Mặc định đọc parquet
            return self.spark.read.parquet(input_path)

    def save_data(
        self, df: DataFrame, output_path: str, format: str = "parquet"
    ) -> None:
        """
        Lưu dữ liệu vào đường dẫn

        Args:
            df: DataFrame cần lưu
            output_path: Đường dẫn đến thư mục đầu ra
            format: Định dạng lưu (parquet, csv, json)
        """
        if not output_path.startswith("hdfs://"):
            output_path = os.path.abspath(output_path)

        if format == "csv":
            df.write.mode("overwrite").option("header", "true").csv(output_path)
        elif format == "json":
            df.write.mode("overwrite").json(output_path)
        else:
            df.write.mode("overwrite").parquet(output_path)

        print(f"Đã lưu {df.count()} bản ghi vào {output_path}")

    def process(
        self, input_path: str, output_path: str, format: str = "parquet"
    ) -> None:
        """
        Xử lý dữ liệu từ đầu đến cuối

        Args:
            input_path: Đường dẫn đến dữ liệu đầu vào
            output_path: Đường dẫn đến thư mục đầu ra
            format: Định dạng lưu (parquet, csv, json)
        """
        # Đọc dữ liệu
        df = self.load_data(input_path)

        # Làm sạch dữ liệu
        df = self.clean_data(df)

        # Biến đổi dữ liệu
        df = self.transform_data(df)

        # Tạo đặc trưng
        df = self.feature_engineering(df)

        # Lưu dữ liệu
        self.save_data(df, output_path, format)

    def stop(self) -> None:
        """Dừng SparkSession"""
        if self._spark:
            self._spark.stop()
            self._spark = None

    @staticmethod
    def parse_args():
        """
        Parse command line arguments
        """
        parser = argparse.ArgumentParser(description="Xử lý dữ liệu")
        parser.add_argument(
            "--input", required=True, help="Đường dẫn đến dữ liệu đầu vào"
        )
        parser.add_argument(
            "--output", required=True, help="Đường dẫn đến thư mục đầu ra"
        )
        parser.add_argument(
            "--format", default="parquet", help="Định dạng lưu (parquet, csv, json)"
        )
        return parser.parse_args()
