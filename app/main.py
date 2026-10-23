import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from app.model import SimpleModel
import time

app = FastAPI(
    title="AI Model Serving Platform",
    version="0.1.0"
)

# 建立模型（服務啟動時載入一次）
from pathlib import Path

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "simple_model.pth"
)

model = SimpleModel()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )
)
model.eval()


# 定義 API 輸入格式
class PredictionRequest(BaseModel):
    features: list[float] = Field(min_length=3, max_length=3)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "ai-model-serving"
    }


@app.post("/predict")
def predict(request: PredictionRequest):

    x = torch.tensor(
        [request.features],
        dtype=torch.float32
    )

    start = time.perf_counter()

    with torch.inference_mode():
        output = model(x)

    latency_ms = (time.perf_counter() - start) * 1000

    return {
        "prediction": output.item(),
        "model": "simple-linear",
        "inference_latency_ms": round(latency_ms, 3)
    }