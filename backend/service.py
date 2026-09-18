import io 
import time 
import base64
from PIL import Image
from ultralytics import YOLO
from fastapi import HTTPException
from backend.schemas import BoxInfo, ModelInfo, PredictResponse

MODEL_CONFIGS = {
    "fast": {
        "name": "YOLOv8-Nano",
        "path": "weights/model_fast.pt",
        "description": "Light model for fast inference",
        "map_score": 0.87
    },
    "accurate": {
            "name": "YOLOv8-Small",
            "path": "weights/model_precise.pt",
            "description": "Heavy model for precise inference",
            "map_score": 0.92
        }
}

class ModelService:
    def __init__(self):
        self.models={}
        self._load_models()
    def _load_models(self):
        for model_id, config in MODEL_CONFIGS.items():
            print(f"Init {model_id} from {config["path"]}...")
            self.models[model_id] = YOLO(config["path"])
        print("All models load successfully!")
    def get_models_info(self) -> list[ModelInfo]:
        return [
            ModelInfo(
                model_id=m_id,
                name=cfg["name"],
                description=cfg["description"],
                map_score=cfg["map_score"]
            )
            for m_id, cfg in MODEL_CONFIGS.items()
        ]
    def predict(self, model_id: str, image_bytes: bytes, conf_threshold: float = 0.25) -> PredictResponse:

        if model_id not in self.models:
            raise HTTPException(status_code=400, detail=f"Model {model_id} is not found. Available: {list(self.models.keys())}")
        
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        start_time = time.perf_counter()
        model = self.models[model_id]
        results = model.predict(source=image, conf=conf_threshold)
        latency_ms = (time.perf_counter() - start_time)*1000
        result = results[0]
        detections = []

        for box in result.boxes:
            coords = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            detections.append(BoxInfo(
                x1=coords[0],
                x2=coords[1],
                y1=coords[2],
                y2=coords[3],
                confidence=round(conf,3),
                class_name=class_name
            ))

        plotted_bgr = result.plot()
        plotted_rgb = Image.fromarray(plotted_bgr[..., ::-1])
        buffered = io.BytesIO()
        plotted_rgb.save(buffered, format='JPEG')
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return PredictResponse(
            model_name=MODEL_CONFIGS[model_id]["name"],
            inference_time_ms=round(latency_ms,2),
            detections=detections,
            image_base64=img_base64
        )
    
detector_service = ModelService()