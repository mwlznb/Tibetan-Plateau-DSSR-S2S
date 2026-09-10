"""Stage 08 resumable conditional residual diffusion training."""
from __future__ import annotations

import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.diffusion.common import ResidualDataset, build_center_cache, diffusion_schedule, forecast_dataset
from src.diffusion.model import ConditionalResidualDiffusionUNet
from src.training.deterministic import _seed_everything
from src.utils.config import PROJECT_ROOT, load_yaml, resolved_paths
from src.utils.hashing import sha256_file, sha256_json


def _atomic_save(value: dict[str, Any], path: Path) -> None:
    temporary = Path(str(path) + ".working")
    torch.save(value, temporary); os.replace(temporary, path)


def _residual_statistics(dataset: ResidualDataset, mask: np.ndarray, scale: float) -> dict[str, float | int]:
    total = total_sq = 0.0; count = 0
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    valid = torch.from_numpy(mask)[None, None].bool()
    for number, batch in enumerate(loader, start=1):
        residual = (batch["target"] - batch["center"]) * scale
        values = residual[valid.expand_as(residual)].double()
        total += values.sum().item(); total_sq += values.square().sum().item(); count += values.numel()
        if number == 1 or number % 250 == 0:
            print(f"Stage08 residual stats batches={number}/{len(loader)}", flush=True)
    mean = total / count
    std = math.sqrt(max(total_sq / count - mean * mean, 0.0))
    if not np.isfinite([mean, std]).all() or std <= 0:
        raise RuntimeError("Invalid training residual statistics")
    return {"mean_wm2": mean, "std_wm2": std, "points": count,
            "basis": "training deterministic residuals on coverage mask only"}


def _batch_size(model: torch.nn.Module, dataset: ResidualDataset, device: torch.device,
                mask: torch.Tensor, cfg: dict[str, Any], stats: dict[str, Any],
                schedule: dict[str, torch.Tensor]) -> int:
    sample = dataset[0]
    for size in (8, 4, 2, 1):
        try:
            model.zero_grad(set_to_none=True)
            def repeat(name: str) -> torch.Tensor:
                return sample[name].unsqueeze(0).expand(size, *sample[name].shape).contiguous().to(device)
            s2s, context, target, center = repeat("s2s"), repeat("context"), repeat("target"), repeat("center")
            lead = sample["lead"].repeat(size).to(device)
            timestep = torch.zeros(size, dtype=torch.long, device=device)
            residual = ((target - center) * 700.0 - float(stats["mean_wm2"])) / float(stats["std_wm2"])
            noise = torch.randn_like(residual)
            noisy = schedule["alpha_bar"][timestep].sqrt()[:, None, None, None] * residual + \
                (1.0 - schedule["alpha_bar"][timestep]).sqrt()[:, None, None, None] * noise
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(cfg["amp"])):
                prediction = model(noisy, s2s, context, center, lead, timestep)
                loss = ((prediction - noise).square() * mask).sum() / mask.expand_as(prediction).sum()
            loss.backward(); model.zero_grad(set_to_none=True); torch.cuda.empty_cache()
            return size
        except torch.cuda.OutOfMemoryError:
            model.zero_grad(set_to_none=True); torch.cuda.empty_cache()
    raise RuntimeError("Diffusion does not fit even batch size 1")


@torch.no_grad()
def _validate(model: torch.nn.Module, loader: DataLoader, device: torch.device, mask: torch.Tensor,
              cfg: dict[str, Any], stats: dict[str, Any], schedule: dict[str, torch.Tensor]) -> float:
    model.eval(); total = 0.0; count = 0
    generator = torch.Generator(device=device).manual_seed(int(cfg["seed"]) + 8000)
    for batch in loader:
        s2s = batch["s2s"].to(device, non_blocking=True)
        context = batch["context"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        center = batch["center"].to(device, non_blocking=True)
        lead = batch["lead"].to(device, non_blocking=True)
        timestep = torch.randint(0, int(cfg["diffusion_steps_K"]), (len(lead),),
                                 device=device, generator=generator)
        residual = ((target - center) * 700.0 - float(stats["mean_wm2"])) / float(stats["std_wm2"])
        noise = torch.randn(residual.shape, device=device, generator=generator)
        alpha = schedule["alpha_bar"][timestep][:, None, None, None]
        noisy = alpha.sqrt() * residual + (1.0 - alpha).sqrt() * noise
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(cfg["amp"])):
            prediction = model(noisy, s2s, context, center, lead, timestep)
        valid = mask.expand_as(prediction)
        total += ((prediction - noise).square() * valid).sum().item(); count += int(valid.sum().item())
    return total / count


def run() -> None:
    done = PROJECT_ROOT / "state" / "stage_08_diffusion.done"
    if done.exists():
        print("Stage08 already complete; skipping", flush=True); return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Stage08")
    cfg = load_yaml("diffusion.yaml"); paths = resolved_paths(); device = torch.device("cuda")
    _seed_everything(int(cfg["seed"]))
    train_center = build_center_cache("train", device)
    val_center = build_center_cache("val", device)
    train_base, val_base = forecast_dataset("train"), forecast_dataset("val")
    train_data = ResidualDataset(train_base, train_center)
    val_data = ResidualDataset(val_base, val_center)
    stats_path = PROJECT_ROOT / "manifests" / "training_residual_statistics.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
    else:
        stats = _residual_statistics(train_data, train_base.mask, train_base.dssr_scale)
        selection = json.loads((PROJECT_ROOT / "manifests" /
                                "deterministic_model_selection.json").read_text(encoding="utf-8"))
        stats["deterministic_checkpoint_sha256"] = selection["checkpoint_SHA256"]
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    schedule = diffusion_schedule(cfg, device)
    model = ConditionalResidualDiffusionUNet(train_base.s2s_mean.shape[0]).to(device)
    mask = torch.from_numpy(train_base.mask.astype(np.float32))[None, None].to(device)
    batch_size = _batch_size(model, train_data, device, mask, cfg, stats, schedule)
    accumulation = max(1, math.ceil(16 / batch_size))
    generator = torch.Generator().manual_seed(int(cfg["seed"]))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, generator=generator,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]),
                                  weight_decay=float(cfg["weight_decay"]))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg["amp"]))
    checkpoint_root = Path(paths["work_root"]) / "checkpoints" / "residual_diffusion"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    best_path, latest_path = checkpoint_root / "best.pt", checkpoint_root / "latest.pt"
    history_path = PROJECT_ROOT / "outputs" / "metrics" / "diffusion_training.csv"
    start_epoch, best_loss, bad_epochs, history = 0, float("inf"), 0, []
    config_sha256 = sha256_json(cfg)
    residual_statistics_sha256 = sha256_file(stats_path)
    if latest_path.exists():
        state = torch.load(latest_path, map_location=device, weights_only=False)
        if state.get("config_sha256") != config_sha256:
            raise RuntimeError("Refusing diffusion resume with a different configuration")
        if state.get("residual_statistics_sha256") != residual_statistics_sha256:
            raise RuntimeError("Refusing diffusion resume with different residual statistics")
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"]); generator.set_state(state["generator_state"].cpu())
        if "cpu_rng_state" in state:
            torch.set_rng_state(state["cpu_rng_state"].cpu())
        if "cuda_rng_state_all" in state:
            torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng_state_all"]])
        start_epoch = int(state["epoch"]) + 1; best_loss = float(state["best_loss"])
        bad_epochs = int(state["bad_epochs"]); history = list(state["history"])
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(start_epoch, int(cfg["max_epochs"])):
        started = time.perf_counter(); model.train(); optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        for step, batch in enumerate(train_loader):
            s2s = batch["s2s"].to(device, non_blocking=True)
            context = batch["context"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            center = batch["center"].to(device, non_blocking=True)
            lead = batch["lead"].to(device, non_blocking=True)
            timestep = torch.randint(0, int(cfg["diffusion_steps_K"]), (len(lead),), device=device)
            residual = ((target - center) * 700.0 - float(stats["mean_wm2"])) / float(stats["std_wm2"])
            noise = torch.randn_like(residual)
            alpha = schedule["alpha_bar"][timestep][:, None, None, None]
            noisy = alpha.sqrt() * residual + (1.0 - alpha).sqrt() * noise
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(cfg["amp"])):
                prediction = model(noisy, s2s, context, center, lead, timestep)
                valid = mask.expand_as(prediction)
                loss = ((prediction - noise).square() * valid).sum() / valid.sum()
            scaler.scale(loss / accumulation).backward()
            if (step + 1) % accumulation == 0 or step + 1 == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["gradient_clip_norm"]))
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            train_loss += float(loss.detach().item())
        val_loss = _validate(model, val_loader, device, mask, cfg, stats, schedule)
        improved = val_loss < best_loss
        if improved: best_loss, bad_epochs = val_loss, 0
        else: bad_epochs += 1
        row = {"epoch": epoch, "train_epsilon_mse": train_loss / len(train_loader),
               "val_epsilon_mse": val_loss, "epoch_duration_seconds": time.perf_counter() - started,
               "peak_gpu_memory_mib": torch.cuda.max_memory_allocated() / 1024 ** 2,
               "batch_size": batch_size, "accumulation_steps": accumulation,
               "device": torch.cuda.get_device_name(0)}
        history.append(row); pd.DataFrame(history).to_csv(history_path, index=False)
        state = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scaler": scaler.state_dict(), "best_loss": best_loss, "bad_epochs": bad_epochs,
                 "history": history, "generator_state": generator.get_state(), "config": cfg,
                 "config_sha256": config_sha256,
                 "residual_statistics_sha256": residual_statistics_sha256,
                 "cpu_rng_state": torch.get_rng_state(),
                 "cuda_rng_state_all": [value.cpu() for value in torch.cuda.get_rng_state_all()],
                 "batch_size": batch_size, "accumulation_steps": accumulation}
        _atomic_save(state, latest_path)
        if improved: _atomic_save(state, best_path)
        print(f"Stage08 epoch={epoch} train={row['train_epsilon_mse']:.6f} "
              f"val={val_loss:.6f} best={best_loss:.6f} bad={bad_epochs}", flush=True)
        if bad_epochs >= int(cfg["early_stopping_patience"]): break
    if not best_path.exists(): raise RuntimeError("Diffusion best checkpoint missing")
    project_checkpoint = PROJECT_ROOT / "checkpoints" / "residual_diffusion_best.pt"
    project_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if not project_checkpoint.exists() or sha256_file(project_checkpoint) != sha256_file(best_path):
        shutil.copy2(best_path, project_checkpoint)
    report = {"stage": 8, "status": "done", "best_validation_epsilon_mse": best_loss,
              "epochs_run": len(history), "checkpoint": str(project_checkpoint),
              "source_checkpoint": str(best_path),
              "checkpoint_sha256": sha256_file(project_checkpoint), "residual_statistics": stats,
              "residual_statistics_sha256": residual_statistics_sha256,
              "config_sha256": config_sha256, "device": torch.cuda.get_device_name(0),
              "amp": bool(cfg["amp"]), "batch_size": batch_size,
              "effective_batch_size": batch_size * accumulation,
              "test_policy": "not loaded or evaluated"}
    report_path = PROJECT_ROOT / "outputs" / "reports" / "stage_08_diffusion_summary.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    done.write_text(json.dumps(report, indent=2), encoding="utf-8")


__all__ = ["run"]
