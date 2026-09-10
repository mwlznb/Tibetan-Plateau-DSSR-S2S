"""Deterministic S2S-to-DSSR U-Net with explicit geographic upsampling."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .geo import GeoBilinearUpsampler


def _groups(channels: int) -> int:
    for value in (8, 4, 2, 1):
        if channels % value == 0:
            return value
    return 1


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.block(value)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.conv = DoubleConv(in_channels + skip_channels, out_channels, dropout)

    def forward(self, value: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        value = F.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat((value, skip), dim=1))


class S2SDSSRUnet(nn.Module):
    def __init__(self, s2s_channels: int, lead_embedding_dim: int = 32, base_channels: int = 32,
                 dropout: float = 0.0, target_type: str = "absolute"):
        super().__init__()
        coarse_lat = torch.linspace(27.0, 37.5, 8)
        coarse_lon = torch.linspace(78.0, 99.0, 15)
        target_lat = torch.linspace(27.0, 37.0, 101)
        target_lon = torch.linspace(78.0, 99.0, 211)
        self.coarse_encoder = DoubleConv(s2s_channels, 64)
        self.geo_upsampler = GeoBilinearUpsampler(coarse_lat, coarse_lon, target_lat, target_lon)
        self.context_encoder = DoubleConv(12, 32)
        self.input_conv = DoubleConv(96, base_channels, dropout)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_channels, base_channels * 2, dropout))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_channels * 2, base_channels * 4, dropout))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_channels * 4, base_channels * 8, dropout))
        self.lead_embedding = nn.Embedding(41, lead_embedding_dim)
        self.film = nn.Sequential(
            nn.Linear(lead_embedding_dim, base_channels * 4), nn.SiLU(),
            nn.Linear(base_channels * 4, base_channels * 16),
        )
        self.up2 = UpBlock(base_channels * 8, base_channels * 4, base_channels * 4, dropout)
        self.up1 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2, dropout)
        self.up0 = UpBlock(base_channels * 2, base_channels, base_channels, dropout)
        self.output = nn.Conv2d(base_channels, 1, 1)
        if target_type not in {"absolute", "climatology_residual"}:
            raise ValueError(f"Unsupported target_type={target_type}")
        self.target_type = target_type

    def forward(self, s2s: torch.Tensor, context: torch.Tensor, lead_day: torch.Tensor) -> torch.Tensor:
        coarse = self.geo_upsampler(self.coarse_encoder(s2s))
        high = self.context_encoder(context)
        x0 = self.input_conv(torch.cat((coarse, high), dim=1))
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        gamma_beta = self.film(self.lead_embedding(lead_day.long())).view(x3.shape[0], 2, x3.shape[1], 1, 1)
        x3 = x3 * (1.0 + gamma_beta[:, 0]) + gamma_beta[:, 1]
        value = self.up2(x3, x2)
        value = self.up1(value, x1)
        value = self.up0(value, x0)
        output = self.output(value)
        return F.softplus(output) if self.target_type == "absolute" else output


__all__ = ["S2SDSSRUnet"]
