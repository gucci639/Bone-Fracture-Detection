from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from backend.schemas import PredictResponse, ModelInfo
from backend.service import detector_service
from weights.get_weights import get

get()

app = FastAPI(
    title="Bone Fracture Detection API",
    description="API for detection xray's fractures with different models",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/health", tags=["System"])
def health_check():
    return {'status': 'ok'}

@app.get('/models', response_model=List[ModelInfo], tags=["Models"])
def get_available_models():
    return detector_service.get_models_info()

@app.post('/predict', response_model=PredictResponse, tags=["Inference"])
async def predict(
    file: UploadFile = File(..., description="Xray (JPEG/PNG)"),
    model_id: str = Form("fast", description='Choose model: fast or accurate'),
    confidence: float = Form(0.25, ge=0.01, le=1.0, description="Confidence threshold (0.01-1.0)")   
):
    if file.content_type not in ['image/jpeg', 'image/png', 'image/jpg']:
        raise HTTPException(status_code=415, detail="Support only JPEG and PNG files")

    image_bytes = await file.read()
    response = detector_service.predict(
        model_id=model_id,
        image_bytes=image_bytes,
        conf_threshold=confidence
    )
    
    return response