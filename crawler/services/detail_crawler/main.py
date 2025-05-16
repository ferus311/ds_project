import os
import sys
import time
import signal
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any
from collections import defaultdict

# Thêm thư mục gốc vào sys.path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from common.queue.kafka_client import KafkaConsumer, KafkaProducer
from common.utils.logging_utils import setup_logging
from sources.batdongsan.playwright.detail_crawler import crawl_detail
from playwright.async_api import async_playwright
from dotenv import load_dotenv

load_dotenv()
logger = setup_logging()


class DetailCrawlerService:
    def __init__(self):
        self.running = True
        self.source = os.environ.get("SOURCE", "batdongsan")
        self.max_concurrent = int(os.environ.get("MAX_CONCURRENT", "5"))
        self.batch_size = int(os.environ.get("BATCH_SIZE", "100"))
        self.retry_limit = int(os.environ.get("RETRY_LIMIT", "3"))
        self.retry_delay = int(os.environ.get("RETRY_DELAY", "5"))

        # Kafka clients
        self.consumer = KafkaConsumer(
            ["property-urls"], group_id="detail-crawler-group"
        )
        self.producer = KafkaProducer()

        # Async components
        self.url_queue = asyncio.Queue(maxsize=100)
        self.batch_queue = asyncio.Queue()
        self.semaphore = asyncio.Semaphore(self.max_concurrent)

        # Monitoring data
        self.stats = {
            "processed": 0,
            "successful": 0,
            "failed": 0,
            "retries": 0,
            "start_time": time.time(),
        }
        self.failed_urls = defaultdict(int)

        # Signal handling
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)

    def stop(self, *args):
        logger.info("Stopping Detail Crawler Service...")
        self.running = False
        self.report_stats()

    def report_stats(self):
        """Report crawling statistics"""
        runtime = time.time() - self.stats["start_time"]
        logger.info(f"=== Crawler Statistics ===")
        logger.info(f"Runtime: {runtime:.2f} seconds")
        logger.info(f"Processed: {self.stats['processed']} URLs")
        logger.info(f"Successful: {self.stats['successful']} URLs")
        logger.info(f"Failed: {self.stats['failed']} URLs")
        logger.info(f"Retries: {self.stats['retries']}")

        if self.stats["processed"] > 0:
            rate = self.stats["processed"] / runtime
            logger.info(f"Processing rate: {rate:.2f} URLs/second")
            success_rate = (self.stats["successful"] / self.stats["processed"]) * 100
            logger.info(f"Success rate: {success_rate:.2f}%")

    async def kafka_callback(self, result):
        """Send crawled data to Kafka"""
        result.setdefault("timestamp", datetime.now().isoformat())
        success = self.producer.send("property-data", result)
        if success:
            logger.info(f"[Kafka] Sent: {result.get('url', 'unknown')}")
            self.stats["successful"] += 1
        else:
            logger.error(f"[Kafka] Failed: {result.get('url', 'unknown')}")
            self.stats["failed"] += 1

    async def kafka_consumer_loop(self):
        """Consume URLs from Kafka and add to queue"""
        logger.info("Starting Kafka consumer loop")
        batch = []

        while self.running:
            # Nếu url_queue đầy, đợi và không tiêu thụ Kafka
            if self.url_queue.full():
                logger.warning("[Kafka] URL queue is full. Pausing consumption...")
                await asyncio.sleep(1.0)
                continue

            message = self.consumer.consume(timeout=1.0)
            if message and "url" in message:
                batch.append(message)

                # Khi đủ batch hoặc gần đầy, gửi qua batch queue
                if len(batch) >= self.batch_size:
                    await self.batch_queue.put(batch.copy())
                    batch.clear()
            else:
                if batch:
                    await self.batch_queue.put(batch.copy())
                    batch.clear()
                await asyncio.sleep(0.05)

        if batch:
            await self.batch_queue.put(batch)

    async def batch_processor_loop(self):
        """Process batches of URLs and distribute to worker queue"""
        logger.info("Starting batch processor loop")

        while self.running:
            try:
                batch = await asyncio.wait_for(self.batch_queue.get(), timeout=1.0)
                logger.info(f"Processing batch of {len(batch)} URLs")

                # Add each URL to the worker queue
                for message in batch:
                    await self.url_queue.put(message)

                self.batch_queue.task_done()
            except asyncio.TimeoutError:
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"[Batch Processor] Error: {e}")
                await asyncio.sleep(0.5)

    async def worker_loop(self, playwright, worker_id: int):
        """Process URLs from the queue and crawl details"""
        logger.info(f"Starting worker {worker_id}")

        while self.running:
            try:
                message = await asyncio.wait_for(self.url_queue.get(), timeout=1.0)
                url = message["url"]
                metadata = {k: v for k, v in message.items() if k != "url"}

                try:
                    async with self.semaphore:
                        self.stats["processed"] += 1
                        logger.info(f"[Worker {worker_id}] Processing: {url}")
                        detail_data = await crawl_detail(playwright, url)

                    if detail_data:
                        if detail_data.get("skipped"):
                            logger.info(f"[Worker {worker_id}] Skipped: {url}")
                        else:
                            detail_data.update(metadata)
                            await self.kafka_callback(detail_data)
                    else:
                        retry_count = self.failed_urls[url] + 1
                        self.failed_urls[url] = retry_count

                        if retry_count <= self.retry_limit:
                            logger.warning(
                                f"[Worker {worker_id}] Retry {retry_count}/{self.retry_limit} for: {url}"
                            )
                            self.stats["retries"] += 1
                            await asyncio.sleep(self.retry_delay)
                            await self.url_queue.put(message)
                        else:
                            logger.error(
                                f"[Worker {worker_id}] Max retries reached for: {url}"
                            )
                            self.stats["failed"] += 1

                except Exception as e:
                    logger.error(f"[Worker {worker_id}] Processing error: {e}")
                    self.stats["failed"] += 1

                self.url_queue.task_done()

            except asyncio.TimeoutError:
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"[Worker {worker_id}] Queue error: {e}")
                await asyncio.sleep(0.5)

    async def monitor_loop(self):
        """Regularly report statistics and monitor health"""
        logger.info("Starting monitoring loop")
        report_interval = 60  # Report every minute

        while self.running:
            await asyncio.sleep(report_interval)
            self.report_stats()

    async def run_async(self):
        logger.info(
            f"Service started with concurrency={self.max_concurrent}, batch_size={self.batch_size}"
        )

        async with async_playwright() as playwright:
            # Start all the required tasks
            consumer_task = asyncio.create_task(self.kafka_consumer_loop())
            batch_processor_task = asyncio.create_task(self.batch_processor_loop())
            monitor_task = asyncio.create_task(self.monitor_loop())

            # Start workers with unique IDs
            workers = [
                asyncio.create_task(self.worker_loop(playwright, i))
                for i in range(self.max_concurrent)
            ]

            # Gather all tasks
            all_tasks = [consumer_task, batch_processor_task, monitor_task] + workers
            await asyncio.gather(*all_tasks)

    def run(self):
        try:
            asyncio.run(self.run_async())
        except KeyboardInterrupt:
            logger.info("Interrupted")
        except Exception as e:
            logger.error(f"Fatal error: {e}")
        finally:
            self.consumer.close()
            self.report_stats()
            logger.info("Service stopped")


def main():
    DetailCrawlerService().run()


if __name__ == "__main__":
    main()
