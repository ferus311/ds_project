#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import json
import shutil
import logging
from typing import Dict, List, Tuple, Any

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description="So sánh và chọn mô hình tốt nhất")
    parser.add_argument(
        "--models-dir", required=True, help="Thư mục chứa các mô hình cần so sánh"
    )
    parser.add_argument(
        "--output", required=True, help="Thư mục để lưu mô hình tốt nhất"
    )
    parser.add_argument(
        "--metric", default="rmse", help="Metric để so sánh (rmse, mae, r2)"
    )
    return parser.parse_args()


def load_model_metrics(model_dir: str) -> Dict[str, float]:
    """
    Tải metrics của mô hình

    Args:
        model_dir: Thư mục chứa mô hình

    Returns:
        Dict[str, float]: Metrics của mô hình
    """
    metrics_path = os.path.join(model_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        logger.warning(f"Không tìm thấy file metrics.json trong {model_dir}")
        return {}

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    return metrics


def find_best_model(
    models_dir: str, metric: str = "rmse"
) -> Tuple[str, Dict[str, float]]:
    """
    Tìm mô hình tốt nhất dựa trên metric

    Args:
        models_dir: Thư mục chứa các mô hình
        metric: Metric để so sánh (rmse, mae, r2)

    Returns:
        Tuple[str, Dict[str, float]]: Đường dẫn đến mô hình tốt nhất và metrics của nó
    """
    # Tìm tất cả các thư mục mô hình
    model_dirs = [
        os.path.join(models_dir, d)
        for d in os.listdir(models_dir)
        if os.path.isdir(os.path.join(models_dir, d))
    ]

    if not model_dirs:
        raise ValueError(f"Không tìm thấy mô hình nào trong {models_dir}")

    # Tải metrics của từng mô hình
    models_metrics = {}
    for model_dir in model_dirs:
        metrics = load_model_metrics(model_dir)
        if metrics and metric in metrics:
            models_metrics[model_dir] = metrics

    if not models_metrics:
        raise ValueError(f"Không tìm thấy metrics cho bất kỳ mô hình nào")

    # Tìm mô hình tốt nhất
    if metric == "r2":  # Đối với R2, giá trị càng cao càng tốt
        best_model_dir = max(
            models_metrics.keys(), key=lambda x: models_metrics[x][metric]
        )
    else:  # Đối với RMSE và MAE, giá trị càng thấp càng tốt
        best_model_dir = min(
            models_metrics.keys(), key=lambda x: models_metrics[x][metric]
        )

    return best_model_dir, models_metrics[best_model_dir]


def copy_model(src_dir: str, dst_dir: str) -> None:
    """
    Sao chép mô hình từ thư mục nguồn sang thư mục đích

    Args:
        src_dir: Thư mục nguồn
        dst_dir: Thư mục đích
    """
    # Tạo thư mục đích nếu chưa tồn tại
    os.makedirs(dst_dir, exist_ok=True)

    # Sao chép các file và thư mục
    for item in os.listdir(src_dir):
        s = os.path.join(src_dir, item)
        d = os.path.join(dst_dir, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)

    logger.info(f"Đã sao chép mô hình từ {src_dir} sang {dst_dir}")


def main():
    """
    Hàm main để so sánh và chọn mô hình tốt nhất
    """
    args = parse_args()

    try:
        # Tìm mô hình tốt nhất
        best_model_dir, best_metrics = find_best_model(args.models_dir, args.metric)

        # Hiển thị thông tin
        logger.info(f"Mô hình tốt nhất: {os.path.basename(best_model_dir)}")
        logger.info(f"Metrics: {json.dumps(best_metrics, indent=2)}")

        # Sao chép mô hình tốt nhất
        copy_model(best_model_dir, args.output)

        # Lưu thông tin bổ sung
        model_info = {
            "best_model": os.path.basename(best_model_dir),
            "metrics": best_metrics,
            "selection_metric": args.metric,
            "selection_date": os.environ.get("AIRFLOW_EXECUTION_DATE", ""),
        }

        with open(os.path.join(args.output, "best_model_info.json"), "w") as f:
            json.dump(model_info, f, indent=2)

        logger.info(f"Đã lưu thông tin mô hình tốt nhất vào {args.output}")

    except Exception as e:
        logger.error(f"Lỗi: {e}")
        raise


if __name__ == "__main__":
    main()
