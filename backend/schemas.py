from pydantic import BaseModel
from typing import List

class BoxInfo(BaseModel):
    x1: float
    x2: float
    y1: float
    y2: float
    confidence: float
    class_name: str

class PredictResponse(BaseModel):
    model_name: str
    inference_time_ms: float
    detections: List[BoxInfo]
    image_base64: str  # Картинка с нарисованными рамками для быстрой отрисовки в UI

class ModelInfo(BaseModel):
    model_id: str
    name: str
    description: str
    map_score: float