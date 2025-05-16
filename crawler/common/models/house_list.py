from dataclasses import dataclass

@dataclass
class HouseListItem:
    link: str                     # URL đến trang chi tiết
    title: str = None             # Tiêu đề ngắn của bài đăng
    price: str = None             # Giá rao bán
    area: str = None              # Diện tích
    location: str = None          # Khu vực/quận huyện
    posted_date: str = None       # Ngày đăng tin
