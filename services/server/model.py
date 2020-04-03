from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class CSSEModel(BaseModel):
    id_: str
    file_name: str
    province_state: Optional[str]
    country_region: str
    lat: float
    long: float
    #confirmed, deaths, recovered
    event_type: str
    record_date: str
    number: List[int]
    created_at: datetime
    last_modified: datetime
