#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict, Type, Any, Optional
import logging
import os

from ml.base_model import BaseModel
from ml.price_prediction_model import PricePredictionModel

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ModelFactory:
    """
    Factory để tạo và quản lý các mô hình ML
    """

    # Registry cho các mô hình
    MODEL_REGISTRY: Dict[str, Dict[str, Type[BaseModel]]] = {
        "price_prediction": {
            "random_forest": PricePredictionModel,
            "gbt": PricePredictionModel,
            "linear_regression": PricePredictionModel,
        },
        # Thêm các loại mô hình khác ở đây
    }

    @classmethod
    def create_model(
        cls, model_type: str, model_variant: str = "default", **kwargs
    ) -> BaseModel:
        """
        Tạo một mô hình dựa trên loại và biến thể

        Args:
            model_type: Loại mô hình (price_prediction, etc.)
            model_variant: Biến thể của mô hình (random_forest, gbt, etc.)
            **kwargs: Tham số bổ sung cho mô hình

        Returns:
            BaseModel: Instance của mô hình

        Raises:
            ValueError: Nếu không tìm thấy mô hình phù hợp
        """
        # Lấy model_type và model_variant từ biến môi trường nếu không được chỉ định
        model_type = model_type or os.environ.get("MODEL_TYPE", "price_prediction")
        model_variant = model_variant or os.environ.get(
            "MODEL_VARIANT", "random_forest"
        )

        if model_type not in cls.MODEL_REGISTRY:
            raise ValueError(f"Không hỗ trợ loại mô hình: {model_type}")

        available_variants = cls.MODEL_REGISTRY[model_type]

        if model_variant not in available_variants:
            # Fallback to default if available
            if "default" in available_variants:
                model_variant = "default"
                logger.warning(
                    f"Biến thể '{model_variant}' không tìm thấy cho loại mô hình '{model_type}', sử dụng default"
                )
            else:
                # Use first available variant
                model_variant = next(iter(available_variants.keys()))
                logger.warning(
                    f"Biến thể '{model_variant}' không tìm thấy cho loại mô hình '{model_type}', sử dụng {model_variant}"
                )

        model_class = cls.MODEL_REGISTRY[model_type][model_variant]

        # Truyền model_type vào kwargs nếu là PricePredictionModel
        if model_class == PricePredictionModel and "model_type" not in kwargs:
            kwargs["model_type"] = model_variant

        return model_class(**kwargs)

    @classmethod
    def register_model(
        cls, model_type: str, model_variant: str, model_class: Type[BaseModel]
    ) -> None:
        """
        Đăng ký một mô hình mới

        Args:
            model_type: Loại mô hình
            model_variant: Biến thể của mô hình
            model_class: Class của mô hình
        """
        if model_type not in cls.MODEL_REGISTRY:
            cls.MODEL_REGISTRY[model_type] = {}

        cls.MODEL_REGISTRY[model_type][model_variant] = model_class
        logger.info(f"Đã đăng ký mô hình: {model_type}.{model_variant}")

    @classmethod
    def get_available_models(cls) -> Dict[str, Dict[str, Any]]:
        """
        Lấy danh sách các mô hình có sẵn

        Returns:
            Dict: Danh sách các mô hình
        """
        result = {}

        for model_type, variants in cls.MODEL_REGISTRY.items():
            result[model_type] = list(variants.keys())

        return result


def main():
    """
    Hàm main để kiểm tra ModelFactory
    """
    # Hiển thị các mô hình có sẵn
    available_models = ModelFactory.get_available_models()
    print("Các mô hình có sẵn:")
    for model_type, variants in available_models.items():
        print(f"- {model_type}: {', '.join(variants)}")

    # Tạo một mô hình
    model = ModelFactory.create_model("price_prediction", "random_forest")
    print(f"Đã tạo mô hình: {model.__class__.__name__} với loại: {model.model_type}")


if __name__ == "__main__":
    main()
