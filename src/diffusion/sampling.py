"""Deterministic-seed DDIM sampling and resumable ensemble caches."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import zarr
from numcodecs import Blosc
from torch.utils.data import DataLoader

from src.diffusion.common import ResidualDataset, build_center_cache, diffusion_schedule, forecast_dataset
from src.diffusion.model import ConditionalResidualDiffusionUNet
from src.utils.config import PROJECT_ROOT, load_yaml, resolved_paths
from src.utils.hashing import sha256_file


def build_ddim_timesteps(alpha_bar: torch.Tensor, steps: int,
                         spacing: str = "uniform_t") -> torch.Tensor:
    """Return an exact, unique, strictly descending DDIM subsequence including K-1 and 0."""
    total = int(alpha_bar.numel())
    if not 2 <= int(steps) <= total:
        raise ValueError(f"steps must be in [2, {total}], got {steps}")
    if spacing == "uniform_t":
        values = np.linspace(total - 1, 0, int(steps))
    elif spacing == "quadratic_t":
        values = np.linspace(np.sqrt(total - 1), 0.0, int(steps)) ** 2
    elif spacing == "uniform_alpha_bar":
        curve = alpha_bar.detach().float().cpu().numpy()
        targets = np.linspace(float(curve[-1]), float(curve[0]), int(steps))
        values = np.interp(targets, curve[::-1], np.arange(total - 1, -1, -1))
    else:
        raise ValueError(f"Unknown DDIM timestep spacing: {spacing}")
    indices = np.rint(values).astype(np.int64)
    indices[0] = total - 1
    indices[-1] = 0
    # Rounding a dense quadratic schedule can duplicate the terminal zero.
    # Project onto the nearest feasible strictly descending integer sequence.
    for position in range(1, len(indices) - 1):
        upper = indices[position - 1] - 1
        lower = len(indices) - 1 - position
        indices[position] = int(np.clip(indices[position], lower, upper))
    if len(np.unique(indices)) != len(indices) or not np.all(np.diff(indices) < 0):
        raise ValueError(
            f"DDIM spacing {spacing!r} produced duplicate/non-descending timesteps: {indices.tolist()}"
        )
    return torch.as_tensor(indices, device=alpha_bar.device, dtype=torch.long)


def ddim_update(value: torch.Tensor, epsilon: torch.Tensor, current: torch.Tensor,
                previous: torch.Tensor | None, eta: float = 0.0,
                generator: torch.Generator | None = None) -> torch.Tensor:
    """One standard epsilon-prediction DDIM step; ``previous=None`` is terminal x0."""
    x0 = (value - (1.0 - current).sqrt() * epsilon) / current.sqrt()
    if previous is None:
        return x0
    eta_value = float(eta)
    sigma = eta_value * (((1.0 - previous) / (1.0 - current)) *
                         (1.0 - current / previous)).clamp_min(0.0).sqrt()
    direction_scale = (1.0 - previous - sigma.square()).clamp_min(0.0).sqrt()
    if eta_value == 0.0:
        noise = torch.zeros_like(value)
    else:
        noise = torch.randn(value.shape, dtype=value.dtype, device=value.device,
                            generator=generator)
    return previous.sqrt() * x0 + direction_scale * epsilon + sigma * noise


def load_diffusion(device: torch.device) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    cfg = load_yaml("diffusion.yaml"); paths = resolved_paths()
    report = json.loads((PROJECT_ROOT / "outputs" / "reports" /
                         "stage_08_diffusion_summary.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(Path(report["checkpoint"]), map_location=device, weights_only=False)
    channels = forecast_dataset("val").s2s_mean.shape[0]
    model = ConditionalResidualDiffusionUNet(channels).to(device)
    model.load_state_dict(checkpoint["model"]); model.eval()
    stats = json.loads((PROJECT_ROOT / "manifests" /
                        "training_residual_statistics.json").read_text(encoding="utf-8"))
    return model, cfg, stats


@torch.no_grad()
def ddim_residual_members(model: torch.nn.Module, batch: dict[str, torch.Tensor],
                          cfg: dict[str, Any], stats: dict[str, Any], device: torch.device,
                          seed: int, steps: int | None = None,
                          spacing: str = "uniform_t", eta: float = 0.0) -> torch.Tensor:
    members = int(cfg["ensemble_members"])
    steps = int(cfg["ddim_steps"] if steps is None else steps)
    if len(batch["lead"]) != 1:
        raise ValueError("DDIM sampler expects one forecast sample")
    schedule = diffusion_schedule(cfg, device); alpha_bar = schedule["alpha_bar"]
    times = build_ddim_timesteps(alpha_bar, steps, spacing)
    generator = torch.Generator(device=device).manual_seed(seed)
    value = torch.randn((members, 1, 101, 211), device=device, generator=generator)
    s2s = batch["s2s"].to(device).expand(members, -1, -1, -1).contiguous()
    context = batch["context"].to(device).expand(members, -1, -1, -1).contiguous()
    center = batch["center"].to(device).expand(members, -1, -1, -1).contiguous()
    lead = batch["lead"].to(device).expand(members)
    for position, timestep in enumerate(times):
        t = timestep.expand(members)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(cfg["amp"])):
            epsilon = model(value, s2s, context, center, lead, t)
        current = alpha_bar[timestep]
        previous = None if position + 1 == len(times) else alpha_bar[times[position + 1]]
        value = ddim_update(value, epsilon, current, previous, eta=eta, generator=generator)
    return float(stats["mean_wm2"]) + float(stats["std_wm2"]) * value[:, 0].float()


def build_ensemble_cache(split: str, device: torch.device) -> Path:
    if split == "test" and not (PROJECT_ROOT / "state" / "FROZEN_TEST.lock").exists():
        raise RuntimeError("Test ensemble generation is forbidden before FROZEN_TEST.lock")
    if split not in {"val", "test"}: raise ValueError(split)
    paths = resolved_paths(); model, cfg, stats = load_diffusion(device)
    center_path = build_center_cache(split, device)
    dataset = ResidualDataset(forecast_dataset(split), center_path)
    sample_manifest_sha256 = sha256_file(
        PROJECT_ROOT / "manifests" / f"samples_{split}.parquet")
    report = json.loads((PROJECT_ROOT / "outputs" / "reports" /
                         "stage_08_diffusion_summary.json").read_text(encoding="utf-8"))
    root = Path(paths["work_root"]) / "cache" / "probabilistic_predictions" / f"{split}.zarr"
    group = zarr.open_group(str(root), mode="a")
    shape = (len(dataset), int(cfg["ensemble_members"]), 101, 211)
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    if "residual_members" not in group:
        group.create_dataset("residual_members", shape=shape, chunks=(1, shape[1], 101, 211),
                             dtype="f2", compressor=compressor, fill_value=np.nan)
        group.create_dataset("complete", shape=(len(dataset),), chunks=(256,), dtype="bool", fill_value=False)
        group.attrs.update({"diffusion_checkpoint_sha256": report["checkpoint_sha256"],
                            "residual_statistics_sha256": report["residual_statistics_sha256"],
                            "sample_manifest_sha256": sample_manifest_sha256,
                            "ddim_steps": int(cfg["ddim_steps"]), "members": int(cfg["ensemble_members"]),
                            "units": "W m-2", "definition": "unscaled generated deterministic residual"})
    if tuple(group["residual_members"].shape) != shape or group.attrs.get(
            "diffusion_checkpoint_sha256") != report["checkpoint_sha256"] or group.attrs.get(
            "residual_statistics_sha256") != report["residual_statistics_sha256"] or group.attrs.get(
            "sample_manifest_sha256") != sample_manifest_sha256 or int(group.attrs.get(
            "ddim_steps", -1)) != int(cfg["ddim_steps"]):
        raise RuntimeError(f"Incompatible probability cache: {root}")
    complete = group["complete"]
    if np.asarray(complete[:], dtype=bool).all():
        print(
            f"DDIM {split} ensemble complete {len(dataset)}/{len(dataset)}; skipping",
            flush=True,
        )
        return root
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    for index, batch in enumerate(loader):
        if not bool(complete[index]):
            residual = ddim_residual_members(model, batch, cfg, stats, device,
                                             int(cfg["seed"]) + index * 1009)
            group["residual_members"][index] = residual.cpu().numpy().astype(np.float16)
            complete[index] = True
        if index == 0 or (index + 1) % 25 == 0 or index + 1 == len(dataset):
            print(f"DDIM {split} ensemble {index + 1}/{len(dataset)}", flush=True)
    if not np.asarray(complete[:]).all(): raise RuntimeError(f"Incomplete {split} probability cache")
    return root


__all__ = ["build_ddim_timesteps", "build_ensemble_cache", "ddim_residual_members",
           "ddim_update", "load_diffusion"]
