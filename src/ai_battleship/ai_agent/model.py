from typing import override

import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, grid_size: int, in_channels: int = 5, hidden_dim: int = 512):
        super().__init__()
        self.grid_size = grid_size
        self.output_dim = grid_size * grid_size

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),  # [32, H, W]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # [64, H, W]
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * grid_size * grid_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.output_dim),  # Q-values
        )

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C=6, H, W]
        x = self.conv(x)
        x = self.fc(x)
        return x  # [B, H*W]
