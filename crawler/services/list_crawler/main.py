import os
import sys
import time
import signal
import logging
import argparse
from datetime import datetime

# Thêm thư mục gốc vào sys.path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from common.queue.kafka_client import KafkaProducer
from common.utils.logging_utils import setup_logging
from sources.batdongsan.playwright.list_crawler import (
    crawl_listings as playwright_crawl,
)

# from sources.batdongsan.selenium.list_crawler import crawl_listings as selenium_crawl

from dotenv import load_dotenv

load_dotenv()


logger = setup_logging()


class ListCrawlerService:
    def __init__(self):
        self.running = True
        self.source = os.environ.get("SOURCE", "batdongsan")
        self.crawler_type = os.environ.get("CRAWLER_TYPE", "playwright")
        self.max_concurrent = int(os.environ.get("MAX_CONCURRENT", "5"))
        self.start_page = int(os.environ.get("START_PAGE", "1"))
        self.end_page = int(os.environ.get("END_PAGE", "500"))
        self.producer = KafkaProducer()

        # Xử lý tín hiệu để dừng service an toàn
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)

    def stop(self, *args):
        logger.info("Stopping List Crawler Service...")
        self.running = False

    async def crawl_and_publish(self):
        """Crawl danh sách và publish URL vào Kafka theo từng batch"""
        logger.info(f"Starting crawler for pages {self.start_page} to {self.end_page}")

        # Định nghĩa callback để xử lý URLs realtime
        async def kafka_publish_callback(url_batch):
            for url in url_batch:
                message = {
                    "url": url,
                    "source": self.source,
                    "timestamp": datetime.now().isoformat(),
                }
                self.producer.send("property-urls", message)

        # Lựa chọn crawler dựa trên cấu hình
        if self.source == "batdongsan":
            if self.crawler_type == "playwright":
                # Truyền vào callback để xử lý urls realtime
                url_count = await playwright_crawl(
                    self.start_page,
                    self.end_page,
                    self.max_concurrent,
                    kafka_callback=kafka_publish_callback,
                )
            else:
                logger.error(f"Unsupported crawler type: {self.crawler_type}")
                return
        else:
            logger.error(f"Unsupported source: {self.source}")
            return

        logger.info(f"Crawled and published {url_count} URLs to Kafka")
        return url_count

    def run(self, interval=3600):
        """Chạy service với chu kỳ lặp lại (mặc định là 1 giờ)"""
        logger.info(
            f"List Crawler Service started with {self.crawler_type} crawler for {self.source}"
        )

        try:
            while self.running:
                start_time = time.time()

                # Crawl và publish URLs
                import asyncio

                asyncio.run(self.crawl_and_publish())

                # Tính thời gian cần sleep
                elapsed_time = time.time() - start_time
                sleep_time = max(0, interval - elapsed_time)

                if sleep_time > 0 and self.running:
                    logger.info(
                        f"Sleeping for {int(sleep_time)} seconds until next crawl cycle"
                    )
                    # Sleep với kiểm tra định kỳ để có thể dừng sớm
                    for _ in range(int(sleep_time)):
                        if not self.running:
                            break
                        time.sleep(1)

        except KeyboardInterrupt:
            logger.info("Received interrupt signal. Stopping...")
        except Exception as e:
            logger.error(f"Error in List Crawler Service: {e}")
        finally:
            logger.info("List Crawler Service stopped")


def main():
    parser = argparse.ArgumentParser(description="Run List Crawler Service")
    parser.add_argument(
        "--interval", type=int, default=3600, help="Crawl interval in seconds"
    )
    parser.add_argument("--once", action="store_true", help="Crawl once and exit")

    args = parser.parse_args()

    service = ListCrawlerService()

    if args.once:
        # Chạy một lần và thoát
        import asyncio

        asyncio.run(service.crawl_and_publish())
    else:
        # Chạy liên tục với interval
        service.run(interval=args.interval)


if __name__ == "__main__":
    main()
