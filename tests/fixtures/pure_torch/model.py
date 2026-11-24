"""Tiny synthetic pure-PyTorch project (fixture, Tier A expected).

No custom CUDA kernels, no extension build. Only torch + torch.cuda.
"""
import torch
import torch.nn as nn


class TinyNet(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.fc = nn.Linear(width, width)

    def forward(self, x):
        return torch.relu(self.fc(x))


def run(device="cuda"):
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = TinyNet().to(dev)
    x = torch.randn(8, 64, device=dev)
    return model(x).sum().item()
