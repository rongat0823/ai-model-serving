
import time
import statistics

import torch
from app.model import SimpleModel

# 建立模型
model = SimpleModel()
model.eval()

x = torch.tensor([[1.0, 2.0, 3.0]])

# Warmup：先執行 10 次，不計入統計
with torch.inference_mode():
    for _ in range(10):
        model(x)

latencies = []

# 正式測量 100 次
with torch.inference_mode():
    for _ in range(100):
        start = time.perf_counter()

        output = model(x)

        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)

# 排序後計算百分位數（Nearest Rank）
latencies.sort()

def percentile(values, p):
    index = max(0, int((len(values) * p + 99) // 100) - 1)
    return values[index]

print(f"Average: {statistics.mean(latencies):.4f} ms")
print(f"P50:     {percentile(latencies, 50):.4f} ms")
print(f"P95:     {percentile(latencies, 95):.4f} ms")
print(f"P99:     {percentile(latencies, 99):.4f} ms")
