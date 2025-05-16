import asyncio
import random
from playwright.async_api import async_playwright
from common.utils.checkpoint import load_checkpoint, save_checkpoint
from sources.batdongsan.playwright.extractors import extract_list_items

# Cấu hình
CHECKPOINT_FILE = "./checkpoint/batdongsan_list_checkpoint.json"
BATCH_SIZE = 1000
MAX_RETRIES = 3
FLUSH_INTERVAL = 30  # giây

user_agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Safari/605.1.15",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Safari/605.1.15",
]


# ================================
# Crawl 1 trang
# ================================


async def crawl_list_page(playwright, page_number: int):
    retries = MAX_RETRIES
    while retries > 0:
        browser = None
        try:
            browser = await playwright.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent=random.choice(user_agents),
                viewport={"width": 1280, "height": 720},
            )
            page = await context.new_page()
            url = f"https://batdongsan.com.vn/nha-dat-ban/p{page_number}"
            print(f"[List Crawler] Crawling page {page_number}: {url}")
            await page.goto(url, timeout=60000)

            html = await page.content()
            listings = extract_list_items(html)

            if not listings:
                print(f"[List Crawler] Page {page_number} - No data found")
                save_checkpoint(CHECKPOINT_FILE, page_number, success=False)
                return []

            print(f"[List Crawler] Page {page_number} - {len(listings)} listings")
            save_checkpoint(CHECKPOINT_FILE, page_number, success=True)

            await browser.close()
            # Trả về danh sách các đối tượng đã chuyển thành dict
            return [item.__dict__ for item in listings] if listings else []

        except Exception as e:
            print(f"❌ Error on page {page_number}: {e}")
            retries -= 1
            await asyncio.sleep(2)
        finally:
            try:
                if browser:
                    await browser.close()
            except:
                pass

    save_checkpoint(CHECKPOINT_FILE, page_number, success=False)
    return []


# ================================
# Crawl nhiều trang
# ================================
async def crawl_listings(
    start_page=1, end_page=100, max_concurrent=5, force=False, kafka_callback=None
):
    """
    Crawl nhiều trang và xử lý URLs theo callback

    Args:
        start_page: Trang bắt đầu
        end_page: Trang kết thúc
        max_concurrent: Số lượng tác vụ đồng thời
        force: Bỏ qua checkpoint
        kafka_callback: Hàm callback để xử lý URLs khi thu thập (để push lên Kafka)

    Returns:
        int: Số lượng URLs đã xử lý
    """
    checkpoint = load_checkpoint(CHECKPOINT_FILE)
    pages = (
        range(start_page, end_page + 1)
        if force
        else [
            p
            for p in range(start_page, end_page + 1)
            if str(p) not in checkpoint or not checkpoint[str(p)]
        ]
    )

    url_count = 0

    async with async_playwright() as playwright:
        sem = asyncio.Semaphore(max_concurrent)

        async def sem_task(pn):
            nonlocal url_count

            async with sem:
                listings = await crawl_list_page(playwright, pn)

                if listings:
                    # Thu thập URLs và xử lý ngay lập tức từng batch nhỏ
                    urls_batch = []
                    batch_size = 50  # Một batch nhỏ để xử lý

                    for item in listings:
                        if "link" in item and item["link"]:
                            urls_batch.append(item["link"])
                            url_count += 1

                            # Khi đủ batch size, gửi lên Kafka và xóa batch
                            if len(urls_batch) >= batch_size and kafka_callback:
                                await kafka_callback(urls_batch)
                                urls_batch = []

                    # Gửi batch cuối cùng nếu còn
                    if urls_batch and kafka_callback:
                        await kafka_callback(urls_batch)

        tasks = [asyncio.create_task(sem_task(page_number)) for page_number in pages]
        await asyncio.gather(*tasks)

    return url_count

    # ================================
    # CLI entry
    # ================================


async def main():
    url_count = await crawl_listings(
        start_page=args.start, end_page=args.end, force=args.force
    )
    print(f"✅ Total URLs processed: {url_count}")




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=200)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    asyncio.run(main())

    # asyncio.run(crawl_list_pages(args.start, args.end, force=args.force))
