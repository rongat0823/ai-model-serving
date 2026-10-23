import torch
from app.model import SimpleModel

model = SimpleModel()
model.eval()

x = torch.tensor([[1.0, 2.0, 3.0]])

with torch.inference_mode():
    output = model(x)

print("Input:", x)
print("Output:", output)