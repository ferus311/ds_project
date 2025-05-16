import logging
import os
import sys
from datetime import datetime


def setup_logging(name=None, level=None):
    """Thiết lập cấu hình logging

    Args:
        name (str, optional): Tên logger. Mặc định là None (sử dụng root logger).
        level (int, optional): Mức độ logging. Mặc định là None (sử dụng từ env var).

    Returns:
        logging.Logger: Logger đã được cấu hình
    """
    # Xác định level từ biến môi trường hoặc mặc định là INFO
    if level is None:
        level_name = os.environ.get("LOG_LEVEL", "INFO")
        level = getattr(logging, level_name)

    # Tạo logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Xóa các handler hiện tại nếu có
    if logger.hasHandlers():
        logger.handlers.clear()

    # Tạo formatter
    log_format = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    formatter = logging.Formatter(log_format)

    # Tạo console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Tạo file handler (tùy chọn)
    log_to_file = os.environ.get("LOG_TO_FILE", "false").lower() == "true"
    if log_to_file:
        log_dir = os.environ.get("LOG_DIR", "logs")
        os.makedirs(log_dir, exist_ok=True)

        today = datetime.now().strftime("%Y%m%d")
        log_file = f"{log_dir}/{name or 'crawler'}_{today}.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
