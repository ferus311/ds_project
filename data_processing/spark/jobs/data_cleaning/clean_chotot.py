#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    regexp_replace,
    when,
    lit,
    to_date,
    trim,
    lower,
    from_unixtime,
)
from pyspark.sql.types import DoubleType, IntegerType, StringType
import argparse
import os
import re


def parse_args():
    parser = argparse.ArgumentParser(description="Làm sạch dữ liệu Chotot")
    parser.add_argument("--input", required=True, help="Đường dẫn đến dữ liệu đầu vào")
    parser.add_argument("--output", required=True, help="Đường dẫn đến thư mục đầu ra")
    return parser.parse_args()


def clean_price(df):
    """Làm sạch và chuẩn hóa cột giá"""
    return df.withColumn(
        "price_cleaned",
        when(col("price").isNull(), None).otherwise(col("price").cast(DoubleType())),
    )


def clean_area(df):
    """Làm sạch và chuẩn hóa cột diện tích"""
    return df.withColumn(
        "area_cleaned",
        when(col("area").isNull(), None)
        .when(
            col("area").contains("m²") | col("area").contains("m2"),
            regexp_replace(regexp_replace(col("area"), "[^0-9.,]", ""), ",", "."),
        )
        .otherwise(col("area")),
    ).withColumn("area_cleaned", col("area_cleaned").cast(DoubleType()))


def extract_location_info(df):
    """Trích xuất thông tin vị trí"""
    # Chotot thường có thông tin vị trí dưới dạng JSON hoặc đã được phân tách
    if "region_name" in df.columns and "area_name" in df.columns:
        return df.withColumn("city", col("region_name")).withColumn(
            "district", col("area_name")
        )
    else:
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


def clean_property_attributes(df):
    """Làm sạch các thuộc tính của bất động sản"""
    # Xử lý số phòng ngủ
    if "rooms" in df.columns:
        df = df.withColumn(
            "bedrooms_cleaned",
            when(col("rooms").isNull(), None).otherwise(
                col("rooms").cast(IntegerType())
            ),
        )
    else:
        df = df.withColumn("bedrooms_cleaned", lit(None).cast(IntegerType()))

    # Xử lý số phòng tắm
    if "toilets" in df.columns:
        df = df.withColumn(
            "bathrooms_cleaned",
            when(col("toilets").isNull(), None).otherwise(
                col("toilets").cast(IntegerType())
            ),
        )
    else:
        df = df.withColumn("bathrooms_cleaned", lit(None).cast(IntegerType()))

    return df


def convert_timestamp(df):
    """Chuyển đổi timestamp thành datetime"""
    if "list_time" in df.columns:
        df = df.withColumn(
            "post_date",
            when(col("list_time").isNull(), None).otherwise(
                from_unixtime(col("list_time") / 1000).cast("date")
            ),
        )

    return df


def main():
    args = parse_args()

    # Khởi tạo Spark session
    spark = (
        SparkSession.builder.appName("Chotot Data Cleaning")
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
        .getOrCreate()
    )

    # Đọc dữ liệu
    input_path = args.input
    if not input_path.startswith("hdfs://"):
        input_path = os.path.abspath(input_path)

    df = spark.read.parquet(input_path)

    # Làm sạch dữ liệu
    df = clean_price(df)
    df = clean_area(df)
    df = extract_location_info(df)
    df = clean_property_attributes(df)
    df = convert_timestamp(df)

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

    # Chuẩn hóa cột title và description
    if "subject" in df.columns:
        df = df.withColumn("title_cleaned", trim(col("subject")))
    else:
        df = df.withColumn("title_cleaned", trim(col("title")))

    if "body" in df.columns:
        df = df.withColumn("description_cleaned", trim(col("body")))
    else:
        df = df.withColumn("description_cleaned", trim(col("description")))

    # Thêm cột property_type dựa trên thông tin có sẵn
    if "category_name" in df.columns:
        df = df.withColumn(
            "property_type",
            when(
                lower(col("category_name")).contains("chung cư")
                | lower(col("category_name")).contains("căn hộ"),
                "apartment",
            )
            .when(
                lower(col("category_name")).contains("biệt thự")
                | lower(col("category_name")).contains("villa"),
                "villa",
            )
            .when(
                lower(col("category_name")).contains("nhà")
                | lower(col("category_name")).contains("mặt tiền"),
                "townhouse",
            )
            .when(lower(col("category_name")).contains("đất"), "land")
            .otherwise("other"),
        )
    else:
        df = df.withColumn(
            "property_type",
            when(
                lower(col("title_cleaned")).contains("chung cư")
                | lower(col("title_cleaned")).contains("căn hộ"),
                "apartment",
            )
            .when(
                lower(col("title_cleaned")).contains("biệt thự")
                | lower(col("title_cleaned")).contains("villa"),
                "villa",
            )
            .when(
                lower(col("title_cleaned")).contains("nhà phố")
                | lower(col("title_cleaned")).contains("nhà mặt tiền"),
                "townhouse",
            )
            .when(
                lower(col("title_cleaned")).contains("đất nền")
                | lower(col("title_cleaned")).contains("đất thổ cư"),
                "land",
            )
            .otherwise("other"),
        )

    # Lưu dữ liệu đã làm sạch
    output_path = args.output
    if not output_path.startswith("hdfs://"):
        output_path = os.path.abspath(output_path)

    df.write.mode("overwrite").parquet(output_path)

    print(f"Đã làm sạch và lưu {df.count()} bản ghi từ Chotot vào {output_path}")

    spark.stop()


if __name__ == "__main__":
    main()
