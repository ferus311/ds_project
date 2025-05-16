import os
import sys
import time
import signal
import logging
import argparse
from datetime import datetime, timedelta

# Thêm thư mục gốc vào sys.path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from common.queue.kafka_client import KafkaConsumer, KafkaProducer
from common.utils.logging_utils import setup_logging

logger = setup_logging()


class RetryService:
    def __init__(self):
        """Khởi tạo Retry Service"""
        self.running = True
        self.max_retries = int(os.environ.get("MAX_RETRIES", "3"))
        self.retry_delay = int(
            os.environ.get("RETRY_DELAY", "3600")
        )  # Thời gian giữa các lần thử lại (giây)

        # Khởi tạo Kafka consumer và producer
        self.failed_consumer = KafkaConsumer(["failed-urls"], "retry-service-group")
        self.url_producer = KafkaProducer()

        # Lưu trữ URL thất bại và số lần thử
        self.retry_counts = {}
        self.last_retry_time = {}

        # Xử lý tín hiệu để dừng service an toàn
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)

    def stop(self, *args):
        """Dừng service an toàn"""
        logger.info("Stopping Retry Service...")
        self.running = False

    def should_retry(self, url):
        """Kiểm tra xem URL có nên được thử lại hay không"""
        retry_count = self.retry_counts.get(url, 0)
        last_time = self.last_retry_time.get(url, datetime.min)

        # Kiểm tra số lần thử và thời gian trôi qua từ lần cuối
        if retry_count >= self.max_retries:
            return False

        time_passed = (datetime.now() - last_time).total_seconds()
        return time_passed >= self.retry_delay

    def run(self):
        """Chạy retry service"""
        logger.info(
            f"Retry Service started with max_retries={self.max_retries}, retry_delay={self.retry_delay}s"
        )

        try:
            while self.running:
                # Kiểm tra các URL trong hàng đợi thất bại
                failed_url_message = self.failed_consumer.consume(timeout=1.0)

                if failed_url_message:
                    # Xử lý message thất bại
                    try:
                        if (
                            isinstance(failed_url_message, dict)
                            and "url" in failed_url_message
                        ):
                            url = failed_url_message["url"]
                        else:
                            url = failed_url_message

                        # Cập nhật số lần thử
                        self.retry_counts[url] = self.retry_counts.get(url, 0) + 1
                        current_retry = self.retry_counts[url]

                        logger.info(
                            f"Received failed URL: {url}, retry count: {current_retry}/{self.max_retries}"
                        )
                    except Exception as e:
                        logger.error(f"Error processing failed URL: {e}")

                # Thử lại các URL đủ điều kiện
                self.retry_failed_urls()

                # Tránh sử dụng 100% CPU
                time.sleep(1)

        except Exception as e:
            logger.error(f"Unexpected error in Retry Service: {e}")
        finally:
            logger.info("Retry Service stopped")
            self.failed_consumer.close()

    def retry_failed_urls(self):
        """Thử lại các URL thất bại đủ điều kiện"""
        urls_to_remove = []
        current_time = datetime.now()

        for url, retry_count in self.retry_counts.items():
            if retry_count < self.max_retries and self.should_retry(url):
                # Gửi lại URL vào queue để xử lý
                logger.info(
                    f"Retrying URL: {url}, attempt: {retry_count}/{self.max_retries}"
                )

                # Tạo message với metadata
                message = {
                    "url": url,
                    "retry_count": retry_count,
                    "timestamp": current_time.isoformat(),
                }

                # Gửi đến Kafka
                success = self.url_producer.send("property-urls", message)
                if success:
                    self.last_retry_time[url] = current_time

                    # Nếu đã thử quá số lần cho phép, đánh dấu để loại bỏ
                    if retry_count >= self.max_retries:
                        urls_to_remove.append(url)
                        logger.warning(
                            f"URL {url} reached max retries ({self.max_retries}), abandoning"
                        )

        # Xóa URLs đã quá số lần thử
        for url in urls_to_remove:
            if url in self.retry_counts:
                del self.retry_counts[url]

            if url in self.last_retry_time:
                del self.last_retry_time[url]


def main():
    """Khởi động Retry Service"""
    parser = argparse.ArgumentParser(description="Run the Retry Service")
    parser.add_argument(
        "--max_retries", type=int, default=None, help="Maximum retry attempts"
    )
    parser.add_argument(
        "--retry_delay", type=int, default=None, help="Delay between retries in seconds"
    )

    args = parser.parse_args()

    # Ghi đè biến môi trường nếu có tham số từ command line
    if args.max_retries:
        os.environ["MAX_RETRIES"] = str(args.max_retries)
    if args.retry_delay:
        os.environ["RETRY_DELAY"] = str(args.retry_delay)

    service = RetryService()
    service.run()


if __name__ == "__main__":
    main()
