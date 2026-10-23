import torch
from pathlib import Path

from app.model import SimpleModel

model = SimpleModel()

Path("models").mkdir(exist_ok=True)

torch.save(
    model.state_dict(),
    "models/simple_model.pth"
)

print("Model saved!")