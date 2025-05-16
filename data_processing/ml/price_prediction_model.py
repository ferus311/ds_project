#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, when, isnan, isnull
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    VectorAssembler,
    StringIndexer,
    OneHotEncoder,
    StandardScaler,
)
from pyspark.ml.regression import RandomForestRegressor, GBTRegressor, LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
import os
import sys
import logging
from typing import Dict, List, Any, Optional

# Thêm đường dẫn thư mục gốc vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ml.base_model import BaseModel

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PricePredictionModel(BaseModel):
    """
    Mô hình dự đoán giá bất động sản
    """

    def __init__(
        self,
        model_type: str = "random_forest",
        categorical_cols: Optional[List[str]] = None,
        numerical_cols: Optional[List[str]] = None,
    ):
        super().__init__(app_name="Real Estate Price Prediction")
        self.model_type = model_type

        # Các cột mặc định nếu không được chỉ định
        self.categorical_cols = categorical_cols or [
            "city",
            "district",
            "property_type",
        ]
        self.numerical_cols = numerical_cols or [
            "area_cleaned",
            "bedrooms_cleaned",
            "bathrooms_cleaned",
        ]

        # Cột label
        self.label_col = "price_cleaned"
        self.prediction_col = "prediction"

        # Danh sách các cột đặc trưng sau khi xử lý
        self.feature_cols = []
        self.indexed_cols = [f"{col}_indexed" for col in self.categorical_cols]
        self.encoded_cols = [f"{col}_encoded" for col in self.categorical_cols]
        self.scaled_cols = [f"{col}_scaled" for col in self.numerical_cols]

        # Các stages của pipeline
        self.indexers = []
        self.encoders = []
        self.scalers = []
        self.assembler = None
        self.regressor = None

    def preprocess(self, df: DataFrame) -> DataFrame:
        """
        Tiền xử lý dữ liệu trước khi huấn luyện

        Args:
            df: DataFrame cần tiền xử lý

        Returns:
            DataFrame: DataFrame đã được tiền xử lý
        """
        # Lọc các dòng có giá trị null trong cột label
        df = df.filter(~(isnull(col(self.label_col)) | isnan(col(self.label_col))))

        # Lọc các dòng có giá trị null trong các cột đặc trưng số
        for num_col in self.numerical_cols:
            df = df.filter(~(isnull(col(num_col)) | isnan(col(num_col))))

        # Xử lý giá trị null trong các cột đặc trưng phân loại
        for cat_col in self.categorical_cols:
            df = df.withColumn(
                cat_col,
                when((isnull(col(cat_col)) | isnan(col(cat_col))), "unknown").otherwise(
                    col(cat_col)
                ),
            )

        return df

    def build_pipeline(self) -> Pipeline:
        """
        Xây dựng pipeline cho mô hình

        Returns:
            Pipeline: Pipeline đã được xây dựng
        """
        stages = []

        # StringIndexer cho các cột phân loại
        self.indexers = [
            StringIndexer(inputCol=c, outputCol=f"{c}_indexed", handleInvalid="keep")
            for c in self.categorical_cols
        ]
        stages.extend(self.indexers)

        # OneHotEncoder cho các cột đã được indexed
        self.encoders = [
            OneHotEncoder(
                inputCol=f"{c}_indexed", outputCol=f"{c}_encoded", dropLast=True
            )
            for c in self.categorical_cols
        ]
        stages.extend(self.encoders)

        # StandardScaler cho các cột số
        self.scalers = [
            StandardScaler(
                inputCol=c, outputCol=f"{c}_scaled", withStd=True, withMean=True
            )
            for c in self.numerical_cols
        ]
        stages.extend(self.scalers)

        # VectorAssembler để kết hợp tất cả các đặc trưng
        feature_cols = self.encoded_cols + self.scaled_cols
        self.feature_cols = feature_cols
        self.assembler = VectorAssembler(
            inputCols=feature_cols, outputCol="features", handleInvalid="keep"
        )
        stages.append(self.assembler)

        # Lựa chọn mô hình hồi quy
        if self.model_type == "random_forest":
            self.regressor = RandomForestRegressor(
                featuresCol="features",
                labelCol=self.label_col,
                predictionCol=self.prediction_col,
                numTrees=100,
                maxDepth=10,
                seed=42,
            )
        elif self.model_type == "gbt":
            self.regressor = GBTRegressor(
                featuresCol="features",
                labelCol=self.label_col,
                predictionCol=self.prediction_col,
                maxIter=100,
                maxDepth=5,
                seed=42,
            )
        else:  # linear_regression
            self.regressor = LinearRegression(
                featuresCol="features",
                labelCol=self.label_col,
                predictionCol=self.prediction_col,
                maxIter=100,
                regParam=0.1,
                elasticNetParam=0.8,
            )

        stages.append(self.regressor)

        # Tạo pipeline
        return Pipeline(stages=stages)

    def evaluate(self, df: DataFrame) -> Dict[str, float]:
        """
        Đánh giá mô hình

        Args:
            df: DataFrame để đánh giá

        Returns:
            Dict[str, float]: Các metrics đánh giá
        """
        # Tạo các evaluator
        evaluators = {
            "rmse": RegressionEvaluator(
                labelCol=self.label_col,
                predictionCol=self.prediction_col,
                metricName="rmse",
            ),
            "mae": RegressionEvaluator(
                labelCol=self.label_col,
                predictionCol=self.prediction_col,
                metricName="mae",
            ),
            "r2": RegressionEvaluator(
                labelCol=self.label_col,
                predictionCol=self.prediction_col,
                metricName="r2",
            ),
        }

        # Tính toán các metrics
        metrics = {}
        for name, evaluator in evaluators.items():
            metrics[name] = evaluator.evaluate(df)

        return metrics


def main():
    """
    Hàm main để chạy mô hình
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Huấn luyện mô hình dự đoán giá bất động sản"
    )
    parser.add_argument(
        "--train", required=True, help="Đường dẫn đến dữ liệu huấn luyện"
    )
    parser.add_argument("--test", required=True, help="Đường dẫn đến dữ liệu kiểm thử")
    parser.add_argument("--output", required=True, help="Đường dẫn để lưu mô hình")
    parser.add_argument(
        "--model-type",
        default="random_forest",
        choices=["random_forest", "gbt", "linear_regression"],
        help="Loại mô hình hồi quy",
    )

    args = parser.parse_args()

    # Khởi tạo mô hình
    model = PricePredictionModel(model_type=args.model_type)

    # Đọc dữ liệu
    train_df = model.spark.read.parquet(args.train)
    test_df = model.spark.read.parquet(args.test)

    # Huấn luyện và đánh giá mô hình
    model.train_and_evaluate(train_df, test_df)

    # Lưu mô hình
    model.save_model(args.output)

    # Dừng Spark session
    model.stop()


if __name__ == "__main__":
    main()
