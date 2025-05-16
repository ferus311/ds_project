#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Any

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Merge dữ liệu từ các nguồn bất động sản")
    parser.add_argument("--batdongsan-file", required=True, help="Đường dẫn đến file dữ liệu Batdongsan")
    parser.add_argument("--chotot-file", required=True, help="Đường dẫn đến file dữ liệu Chotot")
    parser.add_argument("--output-file", required=True, help="Đường dẫn đến file đầu ra")
    return parser.parse_args()


def load_json_file(file_path: str) -> List[Dict[str, Any]]:
    """Đọc dữ liệu từ file JSON"""
    try:
        if not os.path.exists(file_path):
            logger.warning(f"File không tồn tại: {file_path}")
            return []

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Đảm bảo dữ liệu là một list
        if not isinstance(data, list):
            data = [data]

        logger.info(f"Đã đọc {len(data)} bản ghi từ {file_path}")
        return data
    except Exception as e:
        logger.error(f"Lỗi khi đọc file {file_path}: {e}")
        return []


def transform_batdongsan_data(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Chuyển đổi dữ liệu Batdongsan sang định dạng thống nhất"""
    transformed_data = []

    for item in data:
        try:
            transformed_item = {
                "source": "batdongsan",
                "source_id": item.get("id", ""),
                "title": item.get("title", ""),
                "description": item.get("description", ""),
                "url": item.get("url", ""),
                "price": item.get("price", ""),
                "price_value": item.get("price_cleaned", 0),
                "area": item.get("area", ""),
                "area_value": item.get("area_cleaned", 0),
                "location": item.get("location", ""),
                "city": item.get("city", ""),
                "district": item.get("district", ""),
                "bedrooms": item.get("bedrooms_cleaned", 0),
                "bathrooms": item.get("bathrooms_cleaned", 0),
                "property_type": item.get("property_type", ""),
                "posted_date": item.get("post_date", ""),
                "crawled_date": datetime.now().strftime("%Y-%m-%d"),
                "raw_data": item
            }
            transformed_data.append(transformed_item)
        except Exception as e:
            logger.error(f"Lỗi khi chuyển đổi dữ liệu Batdongsan: {e}")

    return transformed_data


def transform_chotot_data(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Chuyển đổi dữ liệu Chotot sang định dạng thống nhất"""
    transformed_data = []

    for item in data:
        try:
            transformed_item = {
                "source": "chotot",
                "source_id": item.get("list_id", ""),
                "title": item.get("subject", item.get("title", "")),
                "description": item.get("body", item.get("description", "")),
                "url": f"https://www.chotot.com/{item.get('list_id', '')}",
                "price": str(item.get("price", "")),
                "price_value": item.get("price_cleaned", item.get("price", 0)),
                "area": str(item.get("size", item.get("area", ""))),
                "area_value": item.get("area_cleaned", item.get("size", 0)),
                "location": f"{item.get('area_name', '')}, {item.get('region_name', '')}",
                "city": item.get("region_name", ""),
                "district": item.get("area_name", ""),
                "bedrooms": item.get("rooms", item.get("bedrooms", 0)),
                "bathrooms": item.get("toilets", item.get("bathrooms", 0)),
                "property_type": item.get("category_name", item.get("property_type", "")),
                "posted_date": item.get("date", ""),
                "crawled_date": datetime.now().strftime("%Y-%m-%d"),
                "raw_data": item
            }
            transformed_data.append(transformed_item)
        except Exception as e:
            logger.error(f"Lỗi khi chuyển đổi dữ liệu Chotot: {e}")

    return transformed_data


def merge_data(batdongsan_data: List[Dict[str, Any]], chotot_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Gộp dữ liệu từ các nguồn"""
    # Đơn giản là gộp 2 list
    merged_data = batdongsan_data + chotot_data
    logger.info(f"Đã gộp {len(batdongsan_data)} bản ghi Batdongsan và {len(chotot_data)} bản ghi Chotot")
    return merged_data


def save_json_file(data: List[Dict[str, Any]], file_path: str) -> None:
    """Lưu dữ liệu vào file JSON"""
    try:
        # Đảm bảo thư mục tồn tại
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"Đã lưu {len(data)} bản ghi vào {file_path}")
    except Exception as e:
        logger.error(f"Lỗi khi lưu file {file_path}: {e}")


def main():
    """Hàm chính"""
    args = parse_args()

    # Đọc dữ liệu
    batdongsan_data = load_json_file(args.batdongsan_file)
    chotot_data = load_json_file(args.chotot_file)

    # Chuyển đổi dữ liệu
    transformed_batdongsan = transform_batdongsan_data(batdongsan_data)
    transformed_chotot = transform_chotot_data(chotot_data)

    # Gộp dữ liệu
    merged_data = merge_data(transformed_batdongsan, transformed_chotot)

    # Lưu dữ liệu
    save_json_file(merged_data, args.output_file)

    logger.info("Hoàn thành merge dữ liệu")


if __name__ == "__main__":
    main()
