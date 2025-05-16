#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import Evaluator
import os
import json
import pickle
from typing import Dict, Any, List, Tuple, Optional, Union
import logging

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    Lớp cơ sở cho tất cả các mô hình ML
    """

    def __init__(self, app_name: str = "ML Model"):
        self.app_name = app_name
        self._spark = None
        self.model = None
        self.pipeline = None
        self.metrics = {}
        self.feature_cols = []
        self.label_col = "label"
        self.prediction_col = "prediction"

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
    def preprocess(self, df: DataFrame) -> DataFrame:
        """
        Tiền xử lý dữ liệu trước khi huấn luyện

        Args:
            df: DataFrame cần tiền xử lý

        Returns:
            DataFrame: DataFrame đã được tiền xử lý
        """
        pass

    @abstractmethod
    def build_pipeline(self) -> Pipeline:
        """
        Xây dựng pipeline cho mô hình

        Returns:
            Pipeline: Pipeline đã được xây dựng
        """
        pass

    @abstractmethod
    def evaluate(self, df: DataFrame) -> Dict[str, float]:
        """
        Đánh giá mô hình

        Args:
            df: DataFrame để đánh giá

        Returns:
            Dict[str, float]: Các metrics đánh giá
        """
        pass

    def train(self, train_df: DataFrame) -> PipelineModel:
        """
        Huấn luyện mô hình

        Args:
            train_df: DataFrame để huấn luyện

        Returns:
            PipelineModel: Mô hình đã được huấn luyện
        """
        logger.info("Tiền xử lý dữ liệu huấn luyện")
        processed_df = self.preprocess(train_df)

        logger.info("Xây dựng pipeline")
        self.pipeline = self.build_pipeline()

        logger.info("Huấn luyện mô hình")
        self.model = self.pipeline.fit(processed_df)

        return self.model

    def predict(self, df: DataFrame) -> DataFrame:
        """
        Dự đoán với mô hình đã huấn luyện

        Args:
            df: DataFrame để dự đoán

        Returns:
            DataFrame: DataFrame với kết quả dự đoán
        """
        if self.model is None:
            raise ValueError("Mô hình chưa được huấn luyện")

        processed_df = self.preprocess(df)
        return self.model.transform(processed_df)

    def train_and_evaluate(
        self, train_df: DataFrame, test_df: DataFrame
    ) -> Tuple[PipelineModel, Dict[str, float]]:
        """
        Huấn luyện và đánh giá mô hình

        Args:
            train_df: DataFrame để huấn luyện
            test_df: DataFrame để kiểm thử

        Returns:
            Tuple[PipelineModel, Dict[str, float]]: Mô hình và metrics
        """
        self.train(train_df)
        predictions = self.predict(test_df)
        self.metrics = self.evaluate(predictions)

        logger.info(f"Metrics đánh giá: {self.metrics}")
        return self.model, self.metrics

    def save_model(self, path: str) -> None:
        """
        Lưu mô hình

        Args:
            path: Đường dẫn để lưu mô hình
        """
        if self.model is None:
            raise ValueError("Mô hình chưa được huấn luyện")

        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(path, exist_ok=True)

        # Lưu mô hình
        model_path = os.path.join(path, "model")
        self.model.write().overwrite().save(model_path)

        # Lưu metrics
        metrics_path = os.path.join(path, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=2)

        # Lưu các thông tin khác
        metadata = {
            "feature_cols": self.feature_cols,
            "label_col": self.label_col,
            "prediction_col": self.prediction_col,
            "app_name": self.app_name,
        }
        metadata_path = os.path.join(path, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Đã lưu mô hình vào {path}")

    def load_model(self, path: str) -> PipelineModel:
        """
        Tải mô hình

        Args:
            path: Đường dẫn để tải mô hình

        Returns:
            PipelineModel: Mô hình đã tải
        """
        # Tải mô hình
        model_path = os.path.join(path, "model")
        self.model = PipelineModel.load(model_path)

        # Tải metrics
        metrics_path = os.path.join(path, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                self.metrics = json.load(f)

        # Tải các thông tin khác
        metadata_path = os.path.join(path, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                self.feature_cols = metadata.get("feature_cols", [])
                self.label_col = metadata.get("label_col", "label")
                self.prediction_col = metadata.get("prediction_col", "prediction")

        logger.info(f"Đã tải mô hình từ {path}")
        return self.model

    def stop(self) -> None:
        """Dừng SparkSession"""
        if self._spark:
            self._spark.stop()
            self._spark = None
