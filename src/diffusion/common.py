"""Frozen deterministic-center loading and leakage-safe residual caches."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
import zarr
from numcodecs import Blosc
from torch.utils.data import DataLoader, Dataset

from src.models.deterministic import S2SDSSRUnet
from src.training.deterministic import ForecastDataset
from src.utils.config import PROJECT_ROOT, load_yaml, resolved_paths
from src.utils.hashing import sha256_file


def load_selected_model(device: torch.device) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    manifest = json.loads((PROJECT_ROOT / "manifests" /
                           "deterministic_model_selection.json").read_text(encoding="utf-8"))
    config_path = (PROJECT_ROOT / "outputs" / "stage06_refinement" /
                   manifest["selected_experiment"] / "config.yaml")
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(Path(manifest["checkpoint"]), map_location=device, weights_only=False)
    channels = int(checkpoint.get("C_s2s", 84))
    model = S2SDSSRUnet(channels, int(cfg.get("lead_embedding_dim", 32)),
                        int(cfg.get("base_channels", 32)), float(cfg.get("dropout", 0.0)),
                        str(cfg.get("target_type", "absolute"))).to(device)
    model.load_state_dict(checkpoint["model"]); model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, cfg, manifest


def forecast_dataset(split: str, target_type: str = "absolute") -> ForecastDataset:
    paths = resolved_paths(); data_cfg = load_yaml("data.yaml")
    normalization = json.loads((PROJECT_ROOT / "manifests" /
                                "s2s_normalization.json").read_text(encoding="utf-8"))
    frame = pd.read_parquet(PROJECT_ROOT / "manifests" / "cache_init_dates.parquet")
    indices = dict(zip(frame.init_date.astype(str), frame.cache_init_index.astype(int)))
    climatology = np.load(Path(paths["work_root"]) / "cache" /
                          "climatology_doy_train_float32.npy", mmap_mode="r")
    return ForecastDataset(PROJECT_ROOT / "manifests" / f"samples_{split}.parquet", indices,
                           paths, normalization, float(data_cfg["dssr_scale_wm2"]),
                           climatology=climatology, target_type=target_type)


class ResidualDataset(Dataset):
    def __init__(self, base: ForecastDataset, center_path: Path):
        self.base = base
        self.rows = base.rows
        self.center = zarr.open_group(str(center_path), mode="r")["data"]

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        value = self.base[index]
        value["center"] = torch.from_numpy(
            np.asarray(self.center[index], dtype=np.float32)[None] / self.base.dssr_scale)
        return value


@torch.no_grad()
def build_center_cache(split: str, device: torch.device, batch_size: int = 16) -> Path:
    """Cache only train/validation center fields; Stage 10 builds test separately after lock."""
    if split not in {"train", "val", "test"}:
        raise ValueError(split)
    if split == "test" and not (PROJECT_ROOT / "state" / "FROZEN_TEST.lock").exists():
        raise RuntimeError("Test center cache is forbidden before FROZEN_TEST.lock")
    paths = resolved_paths(); model, cfg, manifest = load_selected_model(device)
    dataset = forecast_dataset(split, "absolute")
    sample_manifest = PROJECT_ROOT / "manifests" / f"samples_{split}.parquet"
    sample_manifest_sha256 = sha256_file(sample_manifest)
    root = Path(paths["work_root"]) / "cache" / "deterministic_predictions" / f"{split}.zarr"
    group = zarr.open_group(str(root), mode="a")
    shape = (len(dataset), 101, 211)
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    if "data" not in group:
        group.create_dataset("data", shape=shape, chunks=(1, 101, 211), dtype="f4",
                             compressor=compressor, fill_value=np.nan)
        group.create_dataset("complete", shape=(len(dataset),), chunks=(256,), dtype="bool", fill_value=False)
        group.attrs.update({"deterministic_checkpoint_sha256": manifest["checkpoint_SHA256"],
                            "sample_manifest_sha256": sample_manifest_sha256,
                            "split": split, "units": "W m-2", "sample_order": f"samples_{split}.parquet"})
    if tuple(group["data"].shape) != shape or group.attrs.get(
            "deterministic_checkpoint_sha256") != manifest["checkpoint_SHA256"] or group.attrs.get(
            "sample_manifest_sha256") != sample_manifest_sha256:
        raise RuntimeError(f"Incompatible deterministic center cache: {root}")
    complete = group["complete"]
    if np.asarray(complete[:], dtype=bool).all():
        print(
            f"Stage08 center cache {split} complete {len(dataset)}/{len(dataset)}; skipping",
            flush=True,
        )
        return root
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    offset = 0
    for batch_number, batch in enumerate(loader, start=1):
        count = len(batch["lead"]); positions = np.arange(offset, offset + count)
        missing = ~np.asarray(complete[positions], dtype=bool)
        if missing.any():
            s2s = batch["s2s"].to(device, non_blocking=True)
            context = batch["context"].to(device, non_blocking=True)
            lead = batch["lead"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(cfg.get("amp", True))):
                center = model(s2s, context, lead)
            if getattr(model, "target_type", "absolute") == "climatology_residual":
                center = center + batch["climatology"].to(device, non_blocking=True)
            values = (center.float() * dataset.dssr_scale).cpu().numpy()[:, 0]
            for local, position in enumerate(positions):
                if missing[local]:
                    group["data"][position] = values[local]; complete[position] = True
        offset += count
        if batch_number == 1 or batch_number % 100 == 0 or offset == len(dataset):
            print(f"Stage08 center cache {split} {offset}/{len(dataset)}", flush=True)
    if not np.asarray(complete[:]).all():
        raise RuntimeError(f"Incomplete {split} deterministic center cache")
    return root


def diffusion_schedule(cfg: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    betas = torch.linspace(float(cfg["beta_start"]), float(cfg["beta_end"]),
                           int(cfg["diffusion_steps_K"]), device=device, dtype=torch.float32)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return {"betas": betas, "alphas": alphas, "alpha_bar": alpha_bar}


__all__ = ["ResidualDataset", "build_center_cache", "diffusion_schedule",
           "forecast_dataset", "load_selected_model"]
