"""Stage 11 clustered uncertainty, sensitivity, spatial and diagnostic analyses."""
from __future__ import annotations

import json
import math
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import zarr

from src.diffusion.common import forecast_dataset
from src.utils.config import PROJECT_ROOT, load_yaml, resolved_paths


def _crps(members: np.ndarray, observation: np.ndarray) -> np.ndarray:
    count = members.shape[0]
    first = np.abs(members - observation[None]).mean(axis=0)
    ordered = np.sort(members, axis=0)
    coefficient = 2 * np.arange(1, count + 1) - count - 1
    return first - (ordered * coefficient[:, None, None]).sum(axis=0) / (count * count)


def _season(month: int) -> str:
    if month in (12, 1, 2): return "DJF"
    if month in (3, 4, 5): return "MAM"
    if month in (6, 7, 8): return "JJA"
    return "SON"


def _moving_block_ci(values: np.ndarray, dates: pd.DatetimeIndex, replicates: int,
                     seed: int) -> tuple[float, float, int]:
    cadence = max(float(np.median(np.diff(dates.values).astype("timedelta64[D]").astype(float))), 1.0)
    block = max(1, int(math.ceil(40.0 / cadence)))
    rng = np.random.default_rng(seed); n = len(values); estimates = np.empty(replicates)
    for replicate in range(replicates):
        sampled: list[int] = []
        while len(sampled) < n:
            start = int(rng.integers(0, n))
            sampled.extend((start + offset) % n for offset in range(block))
        estimates[replicate] = values[np.asarray(sampled[:n])].mean()
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high), block


def run() -> None:
    done = PROJECT_ROOT / "state" / "stage_11_uncertainty.done"
    if done.exists(): print("Stage11 already complete; skipping", flush=True); return
    cfg = load_yaml("evaluation.yaml"); paths = resolved_paths()
    lock = json.loads((PROJECT_ROOT / "state" / "FROZEN_TEST.lock").read_text(encoding="utf-8"))
    dataset = forecast_dataset("test"); mask = dataset.mask
    cache_root = Path(paths["work_root"]) / "cache"
    center = zarr.open_group(str(cache_root / "deterministic_predictions" / "test.zarr"), mode="r")["data"]
    residual = zarr.open_group(str(cache_root / "probabilistic_predictions" / "test.zarr"), mode="r")["residual_members"]
    s_res = s_rec = float(load_yaml("evaluation.yaml")["residual_scale"])
    spatial_raw = np.zeros((101, 211)); spatial_rec = np.zeros((101, 211)); spatial_count = np.zeros((101, 211))
    rank = np.zeros(int(load_yaml("diffusion.yaml")["ensemble_members"]) + 1, dtype=np.int64)
    per_init: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    seasonal: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0.0])
    cases: list[dict[str, Any]] = []
    subset_specs: list[tuple[int, int, np.ndarray]] = []
    for size in cfg["member_sensitivity_sizes"]:
        seeds = cfg["member_sensitivity_seeds"] if int(size) < 30 else [cfg["member_sensitivity_seeds"][0]]
        for seed in seeds:
            subset = np.random.default_rng(int(seed)).choice(30, size=int(size), replace=False)
            subset_specs.append((int(size), int(seed), subset))
    sensitivity: dict[tuple[int, int, str], list[float]] = defaultdict(lambda: [0.0, 0.0])
    drift = {"raw_sum": 0.0, "raw_sq": 0.0, "rec_sum": 0.0, "rec_sq": 0.0, "points": 0}
    checkpoint_path = (PROJECT_ROOT / "outputs" / "metrics" /
                       "stage11_diagnostics_checkpoint.pkl")
    signature = {"samples": len(dataset), "center_path": str(cache_root / "deterministic_predictions" / "test.zarr"),
                 "residual_path": str(cache_root / "probabilistic_predictions" / "test.zarr"),
                 "s_res": s_res, "s_rec": s_rec,
                 "subsets": [(size, seed, subset.tolist()) for size, seed, subset in subset_specs]}
    start_index = 0
    if checkpoint_path.exists():
        with checkpoint_path.open("rb") as stream:
            state = pickle.load(stream)
        if state.get("signature") != signature:
            raise RuntimeError("Incompatible Stage11 diagnostics checkpoint")
        start_index = int(state["next_index"])
        spatial_raw, spatial_rec = state["spatial_raw"], state["spatial_rec"]
        spatial_count, rank = state["spatial_count"], state["rank"]
        per_init.update(state["per_init"]); seasonal.update(state["seasonal"])
        cases = state["cases"]; sensitivity.update(state["sensitivity"])
        drift = state["drift"]
        print(f"Stage11 diagnostics resume {start_index}/{len(dataset)}", flush=True)
    for index in range(start_index, len(dataset)):
        sample = dataset[index]; row = dataset.rows.iloc[index]
        observation = sample["target"].numpy()[0] * dataset.dssr_scale
        center_value = np.asarray(center[index], dtype=np.float64)
        draws = np.asarray(residual[index], dtype=np.float64)
        centered = draws - draws.mean(axis=0, keepdims=True)
        raw_members = np.maximum(center_value[None] + s_res * draws, 0.0)
        rec_members = np.maximum(center_value[None] + s_rec * centered, 0.0)
        raw_crps, rec_crps = _crps(raw_members, observation), _crps(rec_members, observation)
        spatial_raw[mask] += raw_crps[mask]; spatial_rec[mask] += rec_crps[mask]
        spatial_count[mask] += 1
        init_text = str(row.init_date)[:10]; valid_points = int(mask.sum())
        per_init[init_text][0] += raw_crps[mask].sum(); per_init[init_text][1] += rec_crps[mask].sum()
        per_init[init_text][2] += valid_points
        season = _season(pd.Timestamp(row.target_date_bjt).month)
        seasonal[(season, "residual")][0] += raw_crps[mask].sum()
        seasonal[(season, "residual")][1] += valid_points
        seasonal[(season, "recentered")][0] += rec_crps[mask].sum()
        seasonal[(season, "recentered")][1] += valid_points
        ranks = (rec_members[:, mask] < observation[mask][None]).sum(axis=0)
        rank += np.bincount(ranks, minlength=len(rank))
        raw_delta = raw_members.mean(axis=0)[mask] - center_value[mask]
        rec_delta = rec_members.mean(axis=0)[mask] - center_value[mask]
        drift["raw_sum"] += raw_delta.sum(); drift["raw_sq"] += np.square(raw_delta).sum()
        drift["rec_sum"] += rec_delta.sum(); drift["rec_sq"] += np.square(rec_delta).sum()
        drift["points"] += valid_points
        cases.append({"sample_id": row.sample_id, "init_date": init_text, "lead_day": int(row.lead_day),
                      "raw_crps": raw_crps[mask].mean(), "recentered_crps": rec_crps[mask].mean(),
                      "difference_rec_minus_raw": (rec_crps[mask] - raw_crps[mask]).mean()})
        for size, seed, subset in subset_specs:
            chosen = draws[subset]; chosen_centered = chosen - chosen.mean(axis=0, keepdims=True)
            pairs = (("residual", np.maximum(center_value[None] + s_res * chosen, 0.0)),
                     ("recentered", np.maximum(center_value[None] + s_rec * chosen_centered, 0.0)))
            for method, members in pairs:
                value = _crps(members, observation)[mask]
                sensitivity[(size, seed, method)][0] += value.sum()
                sensitivity[(size, seed, method)][1] += valid_points
        if index == 0 or (index + 1) % 25 == 0 or index + 1 == len(dataset):
            print(f"Stage11 diagnostics {index + 1}/{len(dataset)}", flush=True)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(str(checkpoint_path) + ".working")
            with temporary.open("wb") as stream:
                pickle.dump({"signature": signature, "next_index": index + 1,
                             "spatial_raw": spatial_raw, "spatial_rec": spatial_rec,
                             "spatial_count": spatial_count, "rank": rank,
                             "per_init": dict(per_init), "seasonal": dict(seasonal),
                             "cases": cases, "sensitivity": dict(sensitivity),
                             "drift": drift}, stream, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temporary, checkpoint_path)
    metrics_root = PROJECT_ROOT / "outputs" / "metrics"; tables_root = PROJECT_ROOT / "outputs" / "tables"
    raw_map = np.divide(spatial_raw, spatial_count, out=np.full_like(spatial_raw, np.nan), where=spatial_count > 0)
    rec_map = np.divide(spatial_rec, spatial_count, out=np.full_like(spatial_rec, np.nan), where=spatial_count > 0)
    np.savez_compressed(metrics_root / "stage11_spatial_crps.npz",
                        residual_crps=raw_map, recentered_crps=rec_map,
                        difference_rec_minus_raw=rec_map - raw_map, mask=mask)
    init_rows = [{"init_date": key, "residual_crps": value[0] / value[2],
                  "recentered_crps": value[1] / value[2],
                  "difference_rec_minus_raw": (value[1] - value[0]) / value[2]}
                 for key, value in sorted(per_init.items())]
    init_frame = pd.DataFrame(init_rows); init_frame.to_csv(metrics_root / "stage11_crps_by_init.csv", index=False)
    low, high, block = _moving_block_ci(init_frame.difference_rec_minus_raw.to_numpy(),
                                        pd.DatetimeIndex(pd.to_datetime(init_frame.init_date)),
                                        int(cfg["bootstrap_replicates"]), int(cfg["bootstrap_seed"]))
    sensitivity_rows = [{"members": key[0], "subset_seed": key[1], "method": key[2],
                         "crps": value[0] / value[1]} for key, value in sorted(sensitivity.items())]
    pd.DataFrame(sensitivity_rows).to_csv(tables_root / "stage11_member_sensitivity.csv", index=False)
    seasonal_rows = [{"season": key[0], "method": key[1], "crps": value[0] / value[1]}
                     for key, value in sorted(seasonal.items())]
    pd.DataFrame(seasonal_rows).to_csv(tables_root / "stage11_seasonal_evaluation.csv", index=False)
    pd.DataFrame({"rank": np.arange(len(rank)), "count": rank}).to_csv(
        tables_root / "stage11_rank_histogram.csv", index=False)
    terrain = np.load(Path(paths["dssr"]["terrain"]), mmap_mode="r")
    slope = np.asarray(terrain[1], dtype=np.float64)[::-1]
    edges = np.quantile(slope[mask], [0.0, 0.25, 0.5, 0.75, 1.0])
    slope_rows = []
    for index in range(4):
        upper_condition = slope <= edges[index + 1] if index == 3 else slope < edges[index + 1]
        selected = mask & (slope >= edges[index]) & upper_condition
        slope_rows.append({"slope_bin": index + 1, "lower": edges[index], "upper": edges[index + 1],
                           "pixels": int(selected.sum()), "residual_crps": np.nanmean(raw_map[selected]),
                           "recentered_crps": np.nanmean(rec_map[selected])})
    pd.DataFrame(slope_rows).to_csv(tables_root / "stage11_slope_evaluation.csv", index=False)
    case_frame = pd.DataFrame(cases).sort_values("difference_rec_minus_raw")
    representatives = pd.concat((case_frame.head(10), case_frame.tail(10))).drop_duplicates("sample_id")
    representatives.to_csv(tables_root / "stage11_representative_cases.csv", index=False)
    points = int(drift["points"])
    report = {"stage": 11, "status": "done", "bootstrap": {
                  "cluster": "init_date", "replicates": int(cfg["bootstrap_replicates"]),
                  "moving_block_length_init_dates": block,
                  "mean_crps_difference_rec_minus_raw": float(init_frame.difference_rec_minus_raw.mean()),
                  "confidence_interval_95": [low, high]},
              "center_drift": {"raw_mean_wm2": drift["raw_sum"] / points,
                               "raw_rmse_wm2": math.sqrt(drift["raw_sq"] / points),
                               "recentered_mean_wm2": drift["rec_sum"] / points,
                               "recentered_rmse_wm2": math.sqrt(drift["rec_sq"] / points)},
              "cma_auxiliary": "NOT EXECUTED — no frozen CMA source path is configured",
              "outputs": ["stage11_spatial_crps.npz", "stage11_crps_by_init.csv",
                          "stage11_member_sensitivity.csv", "stage11_seasonal_evaluation.csv",
                          "stage11_slope_evaluation.csv", "stage11_rank_histogram.csv",
                          "stage11_representative_cases.csv"]}
    report_path = PROJECT_ROOT / "outputs" / "reports" / "stage_11_uncertainty_summary.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    done.write_text(json.dumps(report, indent=2), encoding="utf-8")
    checkpoint_path.unlink(missing_ok=True)


__all__ = ["run"]
