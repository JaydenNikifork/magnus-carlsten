import torch
import time

# Simulate training speed
dummy_model = torch.nn.Sequential(
    torch.nn.Linear(768, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 1)
)
optimizer = torch.optim.AdamW(dummy_model.parameters())
loss_fn = torch.nn.MSELoss()

# Time 1000 training steps
start = time.time()
for _ in range(1000):
    x = torch.randn(128, 768)  # batch of 128
    y = torch.randn(128, 1)
    loss = loss_fn(dummy_model(x), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

elapsed = time.time() - start
print(f"1000 steps took {elapsed:.1f}s")
print(f"Estimated time for 500k samples (5 epochs): {elapsed * 500_000 / 128 / 1000 / 60:.1f} minutes")