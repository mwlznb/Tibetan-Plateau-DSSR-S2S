"""Conditional residual diffusion U-Net with separate lead and diffusion-time embeddings."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from src.models.deterministic import DoubleConv, UpBlock
from src.models.geo import GeoBilinearUpsampler


def timestep_embedding(timestep: torch.Tensor, dimension: int) -> torch.Tensor:
    half = dimension // 2
    frequencies = torch.exp(-math.log(10000.0) * torch.arange(
        half, device=timestep.device, dtype=torch.float32) / max(half - 1, 1))
    angles = timestep.float()[:, None] * frequencies[None]
    value = torch.cat((torch.sin(angles), torch.cos(angles)), dim=1)
    return F.pad(value, (0, dimension - value.shape[1]))


class ConditionalResidualDiffusionUNet(nn.Module):
    def __init__(self, s2s_channels: int, base_channels: int = 32, embedding_dim: int = 64):
        super().__init__()
        coarse_lat = torch.linspace(27.0, 37.5, 8)
        coarse_lon = torch.linspace(78.0, 99.0, 15)
        target_lat = torch.linspace(27.0, 37.0, 101)
        target_lon = torch.linspace(78.0, 99.0, 211)
        self.coarse_encoder = DoubleConv(s2s_channels, 32)
        self.geo_upsampler = GeoBilinearUpsampler(coarse_lat, coarse_lon, target_lat, target_lon)
        self.context_encoder = DoubleConv(13, 32)
        self.input_conv = DoubleConv(65, base_channels)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_channels, base_channels * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_channels * 2, base_channels * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_channels * 4, base_channels * 8))
        self.lead_embedding = nn.Embedding(41, embedding_dim)
        self.time_mlp = nn.Sequential(nn.Linear(embedding_dim, embedding_dim * 2), nn.SiLU(),
                                      nn.Linear(embedding_dim * 2, embedding_dim))
        self.film = nn.Sequential(nn.Linear(embedding_dim * 2, base_channels * 8), nn.SiLU(),
                                  nn.Linear(base_channels * 8, base_channels * 16))
        self.up2 = UpBlock(base_channels * 8, base_channels * 4, base_channels * 4)
        self.up1 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up0 = UpBlock(base_channels * 2, base_channels, base_channels)
        self.output = nn.Conv2d(base_channels, 1, 1)

    def forward(self, noisy_residual: torch.Tensor, s2s: torch.Tensor, context: torch.Tensor,
                center: torch.Tensor, lead_day: torch.Tensor, diffusion_timestep: torch.Tensor) -> torch.Tensor:
        coarse = self.geo_upsampler(self.coarse_encoder(s2s))
        high = self.context_encoder(torch.cat((context, center), dim=1))
        x0 = self.input_conv(torch.cat((noisy_residual, coarse, high), dim=1))
        x1, x2 = self.down1(x0), None
        x2 = self.down2(x1); x3 = self.down3(x2)
        lead = self.lead_embedding(lead_day.long())
        diffusion = self.time_mlp(timestep_embedding(diffusion_timestep, lead.shape[1]))
        gamma_beta = self.film(torch.cat((lead, diffusion), dim=1)).view(
            x3.shape[0], 2, x3.shape[1], 1, 1)
        x3 = x3 * (1.0 + gamma_beta[:, 0]) + gamma_beta[:, 1]
        value = self.up2(x3, x2); value = self.up1(value, x1); value = self.up0(value, x0)
        return self.output(value)


__all__ = ["ConditionalResidualDiffusionUNet", "timestep_embedding"]
