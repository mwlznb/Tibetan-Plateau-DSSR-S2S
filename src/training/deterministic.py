"""Stage 06: resumable CUDA/AMP training of the frozen-schema D2 model."""
from __future__ import annotations

import json
import math
import os
import random
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import zarr
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.baselines.run import climatology_bin
from src.models.deterministic import S2SDSSRUnet
from src.utils.config import PROJECT_ROOT, load_yaml, resolved_paths
from src.utils.stage import StageBlocked


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def _fast_cache_ready(cache_root: Path, source_name: str) -> bool:
    """Only expose a fast cache after its verified completion metadata is durable."""
    stem = Path(source_name).stem
    fast_path = cache_root / f"{stem}_fast_float32.npy"
    metadata_path = cache_root / f"{stem}_fast_metadata.json"
    if not fast_path.exists() or not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    return metadata.get("status") == "complete"


class ForecastDataset(Dataset):
    def __init__(self, manifest: Path, cache_indices: dict[str, int], paths: dict[str, Any],
                 normalization: dict[str, Any], dssr_scale: float, no_history: bool = False,
                 climatology: np.ndarray | None = None, target_type: str = "absolute",
                 s2s_cache_name: str = "s2s_features.zarr",
                 s2s_channel_indices: list[int] | None = None):
        self.rows = pd.read_parquet(manifest, columns=[
            "sample_id", "init_date", "lead_day", "target_date_bjt", "dssr_target_index",
        ]).reset_index(drop=True)
        self.cache_indices = cache_indices
        cache_root = Path(paths["work_root"]) / "cache"
        s2s_fast = cache_root / f"{Path(s2s_cache_name).stem}_fast_float32.npy"
        history_fast = cache_root / "history_dssr_fast_float32.npy"
        if _fast_cache_ready(cache_root, s2s_cache_name):
            self.s2s = np.load(s2s_fast, mmap_mode="r")
            self.s2s_backend = "npy_memmap"
        else:
            self.s2s = zarr.open_group(str(cache_root / s2s_cache_name), mode="r")["data"]
            self.s2s_backend = "zarr"
        if _fast_cache_ready(cache_root, "history_dssr.zarr"):
            self.history = np.load(history_fast, mmap_mode="r")
            self.history_backend = "npy_memmap"
        else:
            self.history = zarr.open_group(str(cache_root / "history_dssr.zarr"), mode="r")["data"]
            self.history_backend = "zarr"
        self.target = np.load(Path(paths["dssr"]["target"]), mmap_mode="r")
        self.mask = np.asarray(np.load(Path(paths["dssr"]["coverage_mask"]), mmap_mode="r"), dtype=bool)[::-1].copy()
        terrain = np.asarray(np.load(Path(paths["dssr"]["terrain"]), mmap_mode="r")[:4], dtype=np.float32)[:, ::-1].copy()
        for channel in (0, 1):
            mean = float(terrain[channel][self.mask].mean())
            std = float(terrain[channel][self.mask].std())
            terrain[channel] = (terrain[channel] - mean) / max(std, 1e-6)
        terrain[:, ~self.mask] = 0.0
        self.terrain = terrain
        self.solar = np.load(Path(paths["dssr"]["solar_geometry"]), mmap_mode="r")
        self.s2s_channel_indices = s2s_channel_indices
        mean = np.asarray(normalization["mean"], dtype=np.float32)
        std = np.asarray(normalization["std"], dtype=np.float32)
        if s2s_channel_indices is not None:
            mean, std = mean[s2s_channel_indices], std[s2s_channel_indices]
        self.s2s_mean = mean[:, None, None]
        self.s2s_std = std[:, None, None]
        self.dssr_scale = float(dssr_scale)
        self.no_history = no_history
        self.climatology = climatology
        self.target_type = target_type
        if target_type not in {"absolute", "climatology_residual"}:
            raise ValueError(f"Unsupported target_type={target_type}")
        if target_type == "climatology_residual" and climatology is None:
            raise ValueError("Training-only climatology is required for residual targets")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows.iloc[index]
        init_text = str(row.init_date)[:10]
        cache_index = self.cache_indices[init_text]
        lead = int(row.lead_day)
        s2s = np.asarray(self.s2s[cache_index, lead - 1], dtype=np.float32)
        if self.s2s_channel_indices is not None:
            s2s = s2s[self.s2s_channel_indices]
        s2s = (s2s - self.s2s_mean) / self.s2s_std
        history = np.asarray(self.history[cache_index], dtype=np.float32)[:, ::-1].copy() / self.dssr_scale
        if self.no_history:
            history.fill(0.0)
        history[:, ~self.mask] = 0.0
        target_date = date.fromisoformat(str(row.target_date_bjt)[:10])
        solar = np.asarray(self.solar[climatology_bin(target_date), 0:1], dtype=np.float32)[:, ::-1].copy()
        solar[:, ~self.mask] = 0.0
        context = np.concatenate((history, solar, self.terrain), axis=0)
        target = np.asarray(self.target[int(row.dssr_target_index)], dtype=np.float32)[::-1].copy()[None] / self.dssr_scale
        climatology = None
        if self.climatology is not None:
            climatology = np.asarray(self.climatology[climatology_bin(target_date)], dtype=np.float32)[::-1].copy()[None]
            climatology[:, ~self.mask] = 0.0
            climatology /= self.dssr_scale
            if self.target_type == "climatology_residual":
                target = target - climatology
        target[:, ~self.mask] = 0.0
        result = {
            "s2s": torch.from_numpy(s2s), "context": torch.from_numpy(context),
            "target": torch.from_numpy(target), "lead": torch.tensor(lead, dtype=torch.long),
            "sample_id": str(row.sample_id),
        }
        if climatology is not None:
            result["climatology"] = torch.from_numpy(climatology)
        return result


def _masked_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                 gradient_weight: float) -> torch.Tensor:
    weight = mask.expand_as(prediction)
    mse = ((prediction - target).square() * weight).sum() / weight.sum()
    dx_mask = weight[..., :, 1:] * weight[..., :, :-1]
    dy_mask = weight[..., 1:, :] * weight[..., :-1, :]
    dx = ((prediction[..., :, 1:] - prediction[..., :, :-1]) -
          (target[..., :, 1:] - target[..., :, :-1])).abs()
    dy = ((prediction[..., 1:, :] - prediction[..., :-1, :]) -
          (target[..., 1:, :] - target[..., :-1, :])).abs()
    gradient = (dx * dx_mask).sum() / dx_mask.sum() + (dy * dy_mask).sum() / dy_mask.sum()
    return mse + gradient_weight * gradient


@torch.no_grad()
def _validate(model: nn.Module, loader: DataLoader, device: torch.device, mask: torch.Tensor,
              amp: bool, scale: float) -> dict[str, float]:
    model.eval()
    squared = absolute = bias = 0.0
    count = 0
    for batch in loader:
        s2s = batch["s2s"].to(device, non_blocking=True)
        context = batch["context"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        lead = batch["lead"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            prediction = model(s2s, context, lead)
        error = (prediction - target) * scale
        valid = mask.expand_as(error)
        squared += float((error.square() * valid).sum().item())
        absolute += float((error.abs() * valid).sum().item())
        bias += float((error * valid).sum().item())
        count += int(valid.sum().item())
    return {"rmse": math.sqrt(squared / count), "mae": absolute / count, "bias": bias / count}


def _atomic_torch_save(value: dict[str, Any], path: Path) -> None:
    temporary = Path(str(path) + ".working")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _choose_batch_size(model: nn.Module, dataset: ForecastDataset, candidates: list[int],
                       device: torch.device, mask: torch.Tensor, amp: bool, gradient_weight: float) -> int:
    sample = dataset[0]
    for size in candidates:
        try:
            model.zero_grad(set_to_none=True)
            s2s = sample["s2s"].unsqueeze(0).expand(size, -1, -1, -1).contiguous().to(device)
            context = sample["context"].unsqueeze(0).expand(size, -1, -1, -1).contiguous().to(device)
            target = sample["target"].unsqueeze(0).expand(size, -1, -1, -1).contiguous().to(device)
            lead = sample["lead"].repeat(size).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                prediction = model(s2s, context, lead)
                loss = _masked_loss(prediction, target, mask, gradient_weight)
            loss.backward()
            del s2s, context, target, lead, prediction, loss
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            return int(size)
        except torch.cuda.OutOfMemoryError:
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
    raise StageBlocked("No configured batch size fits GPU memory")


def run(force: bool = False) -> tuple[list[Path], list[Path], list[str]]:
    paths = resolved_paths()
    cfg = load_yaml("deterministic.yaml")
    data_cfg = load_yaml("data.yaml")
    stage05_done = PROJECT_ROOT / "state" / "stage_05_run_baselines.done"
    schema_path = PROJECT_ROOT / "manifests" / "feature_schema.json"
    normalization_path = PROJECT_ROOT / "manifests" / "s2s_normalization.json"
    cache_index_path = PROJECT_ROOT / "manifests" / "cache_init_dates.parquet"
    train_path = PROJECT_ROOT / "manifests" / "samples_train.parquet"
    val_path = PROJECT_ROOT / "manifests" / "samples_val.parquet"
    required = [stage05_done, schema_path, normalization_path, cache_index_path, train_path, val_path]
    if not all(path.exists() for path in required):
        raise StageBlocked("Stage05 and frozen cache/schema manifests are required")
    if not torch.cuda.is_available():
        raise StageBlocked("CUDA is required for Stage06")
    device = torch.device("cuda")
    seed = int(cfg["seed"])
    _seed_everything(seed)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
    cache_frame = pd.read_parquet(cache_index_path)
    cache_indices = dict(zip(cache_frame.init_date.astype(str), cache_frame.cache_init_index.astype(int)))
    train_dataset = ForecastDataset(train_path, cache_indices, paths, normalization,
                                    float(data_cfg["dssr_scale_wm2"]))
    val_dataset = ForecastDataset(val_path, cache_indices, paths, normalization,
                                  float(data_cfg["dssr_scale_wm2"]))
    model = S2SDSSRUnet(int(schema["C_s2s"]), int(cfg["lead_embedding_dim"])).to(device)
    mask = torch.from_numpy(train_dataset.mask.astype(np.float32))[None, None].to(device)
    amp = bool(cfg["amp"])
    batch_size = _choose_batch_size(model, train_dataset, [int(v) for v in cfg["batch_size_candidates"]],
                                    device, mask, amp, float(cfg["gradient_loss_weight"]))
    accumulation_steps = max(1, math.ceil(int(cfg["minimum_effective_batch"]) / batch_size))
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator,
                              num_workers=0, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]),
                                  weight_decay=float(cfg["weight_decay"]))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    checkpoint_root = Path(paths["work_root"]) / "checkpoints" / "deterministic_d2"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    latest = checkpoint_root / "latest.pt"
    best = checkpoint_root / "best.pt"
    history_path = PROJECT_ROOT / "outputs" / "metrics" / "deterministic_d2_training.csv"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    start_epoch, best_rmse, bad_epochs = 0, float("inf"), 0
    history_rows: list[dict[str, Any]] = []
    if latest.exists() and not force:
        checkpoint = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        if "data_generator_state" in checkpoint:
            generator.set_state(checkpoint["data_generator_state"].cpu())
        start_epoch = int(checkpoint["epoch"]) + 1
        best_rmse = float(checkpoint["best_rmse"])
        bad_epochs = int(checkpoint["bad_epochs"])
        history_rows = list(checkpoint.get("history", []))
    max_epochs = int(cfg["max_epochs"])
    for epoch in range(start_epoch, max_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        steps = 0
        for step, batch in enumerate(train_loader):
            s2s = batch["s2s"].to(device, non_blocking=True)
            context = batch["context"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            lead = batch["lead"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                prediction = model(s2s, context, lead)
                loss = _masked_loss(prediction, target, mask, float(cfg["gradient_loss_weight"]))
                scaled_loss = loss / accumulation_steps
            scaler.scale(scaled_loss).backward()
            if (step + 1) % accumulation_steps == 0 or step + 1 == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["gradient_clip_norm"]))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            train_loss += float(loss.detach().item())
            steps += 1
        validation = _validate(model, val_loader, device, mask, amp, float(data_cfg["dssr_scale_wm2"]))
        improved = validation["rmse"] < best_rmse
        if improved:
            best_rmse = validation["rmse"]
            bad_epochs = 0
        else:
            bad_epochs += 1
        row = {"epoch": epoch, "train_loss": train_loss / max(steps, 1),
               "val_rmse_wm2": validation["rmse"], "val_mae_wm2": validation["mae"],
               "val_bias_wm2": validation["bias"], "batch_size": batch_size,
               "accumulation_steps": accumulation_steps, "cuda_device": torch.cuda.get_device_name(0)}
        history_rows.append(row)
        pd.DataFrame(history_rows).to_csv(history_path, index=False)
        state = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scaler": scaler.state_dict(), "best_rmse": best_rmse, "bad_epochs": bad_epochs,
                 "history": history_rows, "schema_version": schema["version"], "C_s2s": schema["C_s2s"],
                 "batch_size": batch_size, "accumulation_steps": accumulation_steps, "seed": seed,
                 "data_generator_state": generator.get_state()}
        _atomic_torch_save(state, latest)
        if improved:
            _atomic_torch_save(state, best)
        print(f"Stage06 epoch={epoch} train={row['train_loss']:.6f} val_rmse={validation['rmse']:.4f}", flush=True)
        if bad_epochs >= int(cfg["early_stopping_patience"]):
            break
    if not best.exists():
        raise StageBlocked("Deterministic best checkpoint was not created")
    report_path = PROJECT_ROOT / "outputs" / "reports" / "stage_06_deterministic_summary.json"
    report_path.write_text(json.dumps({
        "experiment": "D2_frozen_schema", "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "cuda": torch.version.cuda, "amp": amp,
        "batch_size": batch_size, "accumulation_steps": accumulation_steps,
        "effective_batch": batch_size * accumulation_steps, "epochs_completed": len(history_rows),
        "best_validation_rmse_wm2": best_rmse, "checkpoint": str(best),
    }, indent=2), encoding="utf-8")
    return [best, latest, history_path, report_path], required, []


__all__ = ["run", "ForecastDataset", "_masked_loss"]
