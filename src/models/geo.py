from __future__ import annotations

import torch
from torch import nn


def geo_sampling_grid(coarse_lat: torch.Tensor, coarse_lon: torch.Tensor,
                      target_lat: torch.Tensor, target_lon: torch.Tensor) -> torch.Tensor:
    """Build align_corners=True grid_sample coordinates from real ascending coordinates."""
    for name, values in (("coarse_lat",coarse_lat),("coarse_lon",coarse_lon),
                         ("target_lat",target_lat),("target_lon",target_lon)):
        if values.ndim != 1 or values.numel() < 2:
            raise ValueError(f"{name} must be one-dimensional with at least two values")
        if not torch.all(values[1:] > values[:-1]):
            raise ValueError(f"{name} must be strictly ascending")
    y = 2*(target_lat-coarse_lat[0])/(coarse_lat[-1]-coarse_lat[0])-1
    x = 2*(target_lon-coarse_lon[0])/(coarse_lon[-1]-coarse_lon[0])-1
    yy,xx=torch.meshgrid(y,x,indexing="ij")
    return torch.stack((xx,yy),dim=-1).unsqueeze(0)


class GeoBilinearUpsampler(nn.Module):
    def __init__(self, coarse_lat: torch.Tensor, coarse_lon: torch.Tensor,
                 target_lat: torch.Tensor, target_lon: torch.Tensor):
        super().__init__()
        geo_sampling_grid(coarse_lat, coarse_lon, target_lat, target_lon)
        lat_hi = torch.searchsorted(coarse_lat, target_lat, right=True).clamp(1, coarse_lat.numel() - 1)
        lon_hi = torch.searchsorted(coarse_lon, target_lon, right=True).clamp(1, coarse_lon.numel() - 1)
        lat_lo, lon_lo = lat_hi - 1, lon_hi - 1
        lat_weight = (target_lat - coarse_lat[lat_lo]) / (coarse_lat[lat_hi] - coarse_lat[lat_lo])
        lon_weight = (target_lon - coarse_lon[lon_lo]) / (coarse_lon[lon_hi] - coarse_lon[lon_lo])
        self.register_buffer("lat_lo", lat_lo)
        self.register_buffer("lat_hi", lat_hi)
        self.register_buffer("lon_lo", lon_lo)
        self.register_buffer("lon_hi", lon_hi)
        self.register_buffer("lat_weight", lat_weight[None, None, :, None])
        self.register_buffer("lon_weight", lon_weight[None, None, None, :])

    def forward(self,x:torch.Tensor)->torch.Tensor:
        low = x.index_select(-2, self.lat_lo)
        high = x.index_select(-2, self.lat_hi)
        latitude_interpolated = low + (high - low) * self.lat_weight
        low = latitude_interpolated.index_select(-1, self.lon_lo)
        high = latitude_interpolated.index_select(-1, self.lon_hi)
        return low + (high - low) * self.lon_weight
