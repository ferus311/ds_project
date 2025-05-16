#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, regexp_replace, when, lit, to_date, trim, lower
from pyspark.sql.types import DoubleType, IntegerType
import os
import sys

# Thêm đường dẫn thư mục gốc vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from jobs.base_data_processor import BaseDataProcessor


class BatdongsanDataProcessor(BaseDataProcessor):
    """
    Bộ xử lý dữ liệu cho nguồn Batdongsan
    """

    def __init__(self):
        super().__init__(app_name="Batdongsan Data Cleaning")

    def clean_price(self, df: DataFrame) -> DataFrame:
        """Làm sạch và chuẩn hóa cột giá"""
        return df.withColumn(
            "price_cleaned",
            when(col("price").isNull(), None)
            .when(
                col("price").contains("tỷ") | col("price").contains("ty"),
                regexp_replace(regexp_replace(col("price"), "[^0-9.,]", ""), ",", ".")
                * 1000000000,
            )
            .when(
                col("price").contains("triệu") | col("price").contains("trieu"),
                regexp_replace(regexp_replace(col("price"), "[^0-9.,]", ""), ",", ".")
                * 1000000,
            )
            .when(col("price").contains("/m²") | col("price").contains("/m2"), None)
            .when(
                col("price").contains("Thỏa thuận")
                | col("price").contains("thoa thuan"),
                None,
            )
            .otherwise(
                regexp_replace(regexp_replace(col("price"), "[^0-9.,]", ""), ",", ".")
            ),
        ).withColumn("price_cleaned", col("price_cleaned").cast(DoubleType()))

    def clean_area(self, df: DataFrame) -> DataFrame:
        """Làm sạch và chuẩn hóa cột diện tích"""
        return df.withColumn(
            "area_cleaned",
            when(col("area").isNull(), None)
            .when(
                col("area").contains("m²") | col("area").contains("m2"),
                regexp_replace(regexp_replace(col("area"), "[^0-9.,]", ""), ",", "."),
            )
            .otherwise(
                regexp_replace(regexp_replace(col("area"), "[^0-9.,]", ""), ",", ".")
            ),
        ).withColumn("area_cleaned", col("area_cleaned").cast(DoubleType()))

    def extract_location_info(self, df: DataFrame) -> DataFrame:
        """Trích xuất thông tin vị trí"""
        return df.withColumn(
            "city",
            when(col("location").isNull(), None).otherwise(
                trim(regexp_replace(col("location"), ",.+", ""))
            ),
        ).withColumn(
            "district",
            when(col("location").isNull(), None)
            .when(
                col("location").contains(","),
                trim(regexp_replace(col("location"), ".+,", "")),
            )
            .otherwise(None),
        )

    def clean_bedrooms(self, df: DataFrame) -> DataFrame:
        """Làm sạch thông tin số phòng ngủ"""
        return df.withColumn(
            "bedrooms_cleaned",
            when(col("bedroom").isNull(), None).otherwise(
                regexp_replace(col("bedroom"), "[^0-9]", "")
            ),
        ).withColumn("bedrooms_cleaned", col("bedrooms_cleaned").cast(IntegerType()))

    def clean_bathrooms(self, df: DataFrame) -> DataFrame:
        """Làm sạch thông tin số phòng tắm"""
        return df.withColumn(
            "bathrooms_cleaned",
            when(col("bathroom").isNull(), None).otherwise(
                regexp_replace(col("bathroom"), "[^0-9]", "")
            ),
        ).withColumn("bathrooms_cleaned", col("bathrooms_cleaned").cast(IntegerType()))

    def clean_data(self, df: DataFrame) -> DataFrame:
        """
        Làm sạch dữ liệu

        Args:
            df: DataFrame cần làm sạch

        Returns:
            DataFrame: DataFrame đã được làm sạch
        """
        df = self.clean_price(df)
        df = self.clean_area(df)
        df = self.extract_location_info(df)
        df = self.clean_bedrooms(df)
        df = self.clean_bathrooms(df)
        return df

    def transform_data(self, df: DataFrame) -> DataFrame:
        """
        Biến đổi dữ liệu

        Args:
            df: DataFrame cần biến đổi

        Returns:
            DataFrame: DataFrame đã được biến đổi
        """
        # Chuẩn hóa cột title và description
        df = df.withColumn("title_cleaned", trim(col("title")))
        df = df.withColumn("description_cleaned", trim(col("description")))

        # Thêm cột property_type dựa trên title
        df = df.withColumn(
            "property_type",
            when(
                lower(col("title")).contains("chung cư")
                | lower(col("title")).contains("căn hộ"),
                "apartment",
            )
            .when(
                lower(col("title")).contains("biệt thự")
                | lower(col("title")).contains("villa"),
                "villa",
            )
            .when(
                lower(col("title")).contains("nhà phố")
                | lower(col("title")).contains("nhà mặt tiền"),
                "townhouse",
            )
            .when(
                lower(col("title")).contains("đất nền")
                | lower(col("title")).contains("đất thổ cư"),
                "land",
            )
            .otherwise("other"),
        )

        return df

    def feature_engineering(self, df: DataFrame) -> DataFrame:
        """
        Tạo các đặc trưng mới

        Args:
            df: DataFrame cần tạo đặc trưng

        Returns:
            DataFrame: DataFrame với các đặc trưng mới
        """
        # Thêm các cột phái sinh
        df = df.withColumn(
            "price_per_m2",
            when(
                (col("price_cleaned").isNotNull())
                & (col("area_cleaned").isNotNull())
                & (col("area_cleaned") > 0),
                col("price_cleaned") / col("area_cleaned"),
            ).otherwise(None),
        )

        # Thêm cột đánh giá giá trị (cao, trung bình, thấp) dựa trên price_per_m2
        df = df.withColumn(
            "price_category",
            when(col("price_per_m2").isNull(), None)
            .when(col("price_per_m2") > 50000000, "high")
            .when(col("price_per_m2") > 30000000, "medium")
            .otherwise("low"),
        )

        return df


def main():
    # Parse command line arguments
    args = BaseDataProcessor.parse_args()

    # Khởi tạo processor
    processor = BatdongsanDataProcessor()

    # Xử lý dữ liệu
    processor.process(args.input, args.output, args.format)

    # Dừng Spark session
    processor.stop()


if __name__ == "__main__":
    main()
