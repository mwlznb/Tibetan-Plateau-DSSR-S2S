"""Leakage-safe deterministic validation metrics shared by Stage 06R trials."""
from __future__ import annotations

import math
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.baselines.run import climatology_bin


class _Moments:
    def __init__(self) -> None:
        self.n = 0
        self.se = self.ae = self.err = 0.0
        self.sp = self.so = self.spp = self.soo = self.spo = 0.0
        self.clim_se = 0.0
        self.sa_p = self.sa_o = self.saa_p = self.saa_o = self.saa_po = 0.0
        self.grad_ae = self.grad_n = 0.0

    def update(self, p: torch.Tensor, o: torch.Tensor, c: torch.Tensor,
               mask: torch.Tensor) -> None:
        valid = mask.expand_as(p).bool()
        pv, ov, cv = p[valid].double(), o[valid].double(), c[valid].double()
        error = pv - ov
        ap, ao = pv - cv, ov - cv
        self.n += pv.numel()
        self.se += error.square().sum().item()
        self.ae += error.abs().sum().item()
        self.err += error.sum().item()
        self.sp += pv.sum().item(); self.so += ov.sum().item()
        self.spp += pv.square().sum().item(); self.soo += ov.square().sum().item()
        self.spo += (pv * ov).sum().item()
        self.clim_se += (cv - ov).square().sum().item()
        self.sa_p += ap.sum().item(); self.sa_o += ao.sum().item()
        self.saa_p += ap.square().sum().item(); self.saa_o += ao.square().sum().item()
        self.saa_po += (ap * ao).sum().item()
        dxm = valid[..., :, 1:] & valid[..., :, :-1]
        dym = valid[..., 1:, :] & valid[..., :-1, :]
        dx = (p[..., :, 1:] - p[..., :, :-1]) - (o[..., :, 1:] - o[..., :, :-1])
        dy = (p[..., 1:, :] - p[..., :-1, :]) - (o[..., 1:, :] - o[..., :-1, :])
        self.grad_ae += dx[dxm].abs().double().sum().item() + dy[dym].abs().double().sum().item()
        self.grad_n += dxm.sum().item() + dym.sum().item()

    @staticmethod
    def _corr(n: int, sx: float, sy: float, sxx: float, syy: float, sxy: float) -> float:
        cov = sxy - sx * sy / n
        vx = sxx - sx * sx / n
        vy = syy - sy * sy / n
        return cov / math.sqrt(max(vx * vy, 1e-30))

    def result(self) -> dict[str, float | int]:
        rmse = math.sqrt(self.se / self.n)
        clim_rmse = math.sqrt(self.clim_se / self.n)
        return {
            "points": self.n, "rmse_wm2": rmse, "mae_wm2": self.ae / self.n,
            "bias_wm2": self.err / self.n,
            "pcc": self._corr(self.n, self.sp, self.so, self.spp, self.soo, self.spo),
            "climatology_rmse_wm2": clim_rmse, "rmse_skill_score": 1.0 - rmse / clim_rmse,
            "anomaly_correlation": self._corr(
                self.n, self.sa_p, self.sa_o, self.saa_p, self.saa_o, self.saa_po),
            "gradient_mae_wm2": self.grad_ae / self.grad_n,
        }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             mask: torch.Tensor, amp: bool, scale: float,
             climatology: np.ndarray) -> pd.DataFrame:
    """Evaluate a sequential validation loader without reading a test manifest."""
    model.eval()
    moments = {lead: _Moments() for lead in range(1, 41)}
    overall = _Moments()
    offset = 0
    rows = loader.dataset.rows
    for batch in loader:
        s2s = batch["s2s"].to(device, non_blocking=True)
        context = batch["context"].to(device, non_blocking=True)
        target_raw = batch["target"].to(device, non_blocking=True)
        lead = batch["lead"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            prediction_raw = model(s2s, context, lead)
        dates = rows.iloc[offset:offset + len(lead)].target_date_bjt.astype(str).str[:10]
        clim_np = np.stack([
            climatology[climatology_bin(date.fromisoformat(value))][::-1].copy()
            for value in dates
        ])[:, None]
        clim = torch.from_numpy(clim_np).to(device)
        if getattr(model, "target_type", "absolute") == "climatology_residual":
            prediction = prediction_raw * scale + clim
            target = target_raw * scale + clim
        else:
            prediction = prediction_raw * scale
            target = target_raw * scale
        for value in torch.unique(lead).tolist():
            selected = lead == int(value)
            moments[int(value)].update(prediction[selected], target[selected], clim[selected], mask)
            overall.update(prediction[selected], target[selected], clim[selected], mask)
        offset += len(lead)
    result: list[dict[str, Any]] = []
    for lead in range(1, 41):
        result.append({"lead_day": lead, **moments[lead].result()})
    result.append({"lead_day": "overall", **overall.result()})
    return pd.DataFrame(result)


__all__ = ["evaluate"]
