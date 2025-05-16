from dataclasses import dataclass


@dataclass
class HouseDetailItem:
    short_description: str = None
    price: str = None
    area: str = None
    price_per_m2: str = None
    coordinates: str = None
    bedroom: str = None
    bathroom: str = None
    house_direction: str = None
    balcony_direction: str = None
    legal_status: str = None
    interior: str = None
    facade_width: str = None
    road_width: str = None
    floor_count: str = None
    title: str = None
    description: str = None
