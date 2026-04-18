from datetime import datetime
from pydantic import BaseModel


class Prediction(BaseModel):
    route_id: int
    office_from_id: int
    timestamp: datetime
    predicted_target_2h: int
