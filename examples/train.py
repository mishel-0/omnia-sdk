"""Minimal training example — copy, paste, run."""
from omnia_sdk.dataset import OmniaDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch

model = nn.Sequential(nn.Flatten(), nn.Linear(512*512, 2)).cuda()
ds = OmniaDataset("./compressed/")
loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=4)

for images, labels in loader:
    out = model(images.cuda())
    loss = nn.CrossEntropyLoss()(out, labels.cuda())
    loss.backward()
    print(f"loss: {loss.item():.4f}")
