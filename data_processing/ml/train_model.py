#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import logging
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, rand

from ml.model_factory import ModelFactory

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description="Huấn luyện và đánh giá mô hình")
    parser.add_argument("--input", required=True, help="Đường dẫn đến dữ liệu đầu vào")
    parser.add_argument("--output", required=True, help="Đường dẫn để lưu mô hình")
    parser.add_argument("--model-type", default="price_prediction", help="Loại mô hình")
    parser.add_argument(
        "--model-variant", default="random_forest", help="Biến thể của mô hình"
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.2, help="Tỷ lệ dữ liệu kiểm thử"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed cho việc chia dữ liệu"
    )
    return parser.parse_args()


def split_data(df: DataFrame, test_ratio: float = 0.2, seed: int = 42) -> tuple:
    """
    Chia dữ liệu thành tập huấn luyện và kiểm thử

    Args:
        df: DataFrame cần chia
        test_ratio: Tỷ lệ dữ liệu kiểm thử
        seed: Seed cho việc chia dữ liệu

    Returns:
        tuple: (train_df, test_df)
    """
    # Chia dữ liệu ngẫu nhiên
    df = df.orderBy(rand(seed=seed))

    # Tính số lượng dòng cho mỗi tập
    count = df.count()
    train_count = int(count * (1 - test_ratio))

    # Chia dữ liệu
    train_df = df.limit(train_count)
    test_df = df.subtract(train_df)

    logger.info(
        f"Đã chia dữ liệu: {train_df.count()} dòng huấn luyện, {test_df.count()} dòng kiểm thử"
    )

    return train_df, test_df


def main():
    """
    Hàm main để huấn luyện và đánh giá mô hình
    """
    args = parse_args()

    # Khởi tạo SparkSession
    spark = (
        SparkSession.builder.appName("Train Real Estate Model")
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
        .getOrCreate()
    )

    try:
        # Đọc dữ liệu
        input_path = args.input
        if not input_path.startswith("hdfs://"):
            input_path = os.path.abspath(input_path)

        df = spark.read.parquet(input_path)
        logger.info(f"Đã đọc {df.count()} bản ghi từ {input_path}")

        # Chia dữ liệu
        train_df, test_df = split_data(df, args.test_ratio, args.seed)

        # Khởi tạo mô hình
        model = ModelFactory.create_model(args.model_type, args.model_variant)

        # Huấn luyện và đánh giá mô hình
        logger.info(
            f"Bắt đầu huấn luyện mô hình {args.model_type}.{args.model_variant}"
        )
        model.train_and_evaluate(train_df, test_df)

        # Lưu mô hình
        output_path = args.output
        if not output_path.startswith("hdfs://"):
            output_path = os.path.abspath(output_path)

        model.save_model(output_path)
        logger.info(f"Đã lưu mô hình vào {output_path}")

        # In metrics
        logger.info("Kết quả đánh giá mô hình:")
        for metric, value in model.metrics.items():
            logger.info(f"- {metric}: {value:.4f}")

    finally:
        # Dừng SparkSession
        spark.stop()


if __name__ == "__main__":
    main()
