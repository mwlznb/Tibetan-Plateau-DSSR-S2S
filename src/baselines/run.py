"""Stage 05: newly computed train-only climatology, persistence and history-mean baselines."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.config import PROJECT_ROOT, resolved_paths
from src.utils.stage import StageBlocked


def climatology_bin(value: date) -> int:
    result = value.timetuple().tm_yday - 1
    if value.month > 2 and not (value.year % 4 == 0 and (value.year % 100 != 0 or value.year % 400 == 0)):
        result += 1
    return result


class MetricAccumulator:
    def __init__(self) -> None:
        self.n = 0
        self.se = 0.0
        self.ae = 0.0
        self.error = 0.0
        self.sum_p = 0.0
        self.sum_o = 0.0
        self.sum_pp = 0.0
        self.sum_oo = 0.0
        self.sum_po = 0.0

    def update(self, prediction: np.ndarray, observation: np.ndarray, mask: np.ndarray) -> None:
        p = np.asarray(prediction[mask], dtype=np.float64)
        o = np.asarray(observation[mask], dtype=np.float64)
        error = p - o
        self.n += p.size
        self.se += float(np.dot(error, error))
        self.ae += float(np.abs(error).sum())
        self.error += float(error.sum())
        self.sum_p += float(p.sum())
        self.sum_o += float(o.sum())
        self.sum_pp += float(np.dot(p, p))
        self.sum_oo += float(np.dot(o, o))
        self.sum_po += float(np.dot(p, o))

    def result(self) -> dict[str, float | int]:
        cov = self.sum_po - self.sum_p * self.sum_o / self.n
        var_p = self.sum_pp - self.sum_p * self.sum_p / self.n
        var_o = self.sum_oo - self.sum_o * self.sum_o / self.n
        pcc = cov / np.sqrt(max(var_p * var_o, 1e-30))
        return {"points": self.n, "rmse": float(np.sqrt(self.se / self.n)),
                "mae": self.ae / self.n, "bias": self.error / self.n, "pcc": float(pcc)}


def _build_climatology(target: np.ndarray, train: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    unique = train[["target_date_bjt", "dssr_target_index"]].drop_duplicates("dssr_target_index")
    sums = np.zeros((366, target.shape[1], target.shape[2]), dtype=np.float64)
    counts = np.zeros(366, dtype=np.int32)
    for row in unique.itertuples(index=False):
        bin_index = climatology_bin(date.fromisoformat(str(row.target_date_bjt)[:10]))
        sums[bin_index] += np.asarray(target[int(row.dssr_target_index)], dtype=np.float64)
        counts[bin_index] += 1
    missing = np.flatnonzero(counts == 0)
    if missing.size:
        raise StageBlocked(f"Train-only climatology has empty calendar bins: {missing.tolist()}")
    return (sums / counts[:, None, None]).astype(np.float32), counts


def _daily_flux_from_cumulative(cumulative: np.ndarray) -> np.ndarray:
    """Convert forecast-start cumulative energy (J m-2) to 24 h mean flux (W m-2)."""
    values = np.asarray(cumulative, dtype=np.float64)
    if values.ndim != 4 or values.shape[1] != 40:
        raise StageBlocked(f"Expected SSRD [member,40,lat,lon], got {values.shape}")
    increments = np.empty_like(values)
    increments[:, 0] = values[:, 0]
    increments[:, 1:] = np.diff(values, axis=1)
    if not np.isfinite(increments).all():
        raise StageBlocked("SSRD daily increments contain non-finite values")
    if float(increments.min()) < -1.0e-3:
        raise StageBlocked("SSRD cumulative energy decreases between forecast leads")
    return increments / 86400.0


def _ssrd_file_map(selected_path: Path) -> dict[str, str]:
    selected = pd.read_parquet(
        selected_path, columns=["variable", "init_date", "source_file"]
    )
    selected = selected.loc[selected["variable"] == "ssrd"].copy()
    selected["init_date"] = selected["init_date"].astype(str).str[:10]
    counts = selected.groupby("init_date")["source_file"].nunique()
    ambiguous = counts[counts != 1]
    if not ambiguous.empty:
        raise StageBlocked(f"SSRD has ambiguous files: {ambiguous.head().to_dict()}")
    unique = selected[["init_date", "source_file"]].drop_duplicates("init_date")
    return dict(zip(unique["init_date"], unique["source_file"]))


def _load_ssrd_ensemble_mean(path: Path, target_grid: dict[str, Any]) -> np.ndarray:
    import xarray as xr

    with xr.open_dataset(path, engine="h5netcdf", decode_cf=False) as dataset:
        if "ssrd" not in dataset:
            raise StageBlocked(f"SSRD variable is missing: {path}")
        array = dataset["ssrd"]
        if "init_time" in array.dims:
            array = array.isel(init_time=0)
        dims = ("number", "step", "latitude", "longitude")
        if not all(name in array.dims for name in dims):
            raise StageBlocked(f"Unexpected SSRD dimensions {array.dims}: {path}")
        steps = np.asarray(dataset["step"].values, dtype=np.float64)
        if steps.shape != (40,) or not np.allclose(steps, np.arange(1, 41) * 24.0):
            raise StageBlocked(f"Unexpected SSRD steps: {steps.tolist()}")
        latitude = np.asarray(dataset["latitude"].values, dtype=np.float64)
        longitude = np.asarray(dataset["longitude"].values, dtype=np.float64)
        values = np.asarray(array.transpose(*dims).values, dtype=np.float64)
    daily_flux = _daily_flux_from_cumulative(values).mean(axis=0)
    field = xr.DataArray(
        daily_flux, dims=("step", "latitude", "longitude"),
        coords={"step": np.arange(1, 41), "latitude": latitude, "longitude": longitude},
    )
    result = np.asarray(field.interp(
        latitude=np.asarray(target_grid["latitude"]),
        longitude=np.asarray(target_grid["longitude"]),
    ).values, dtype=np.float32)
    if result.shape != (40, 101, 211) or not np.isfinite(result).all():
        raise StageBlocked(f"Invalid interpolated SSRD field {result.shape}: {path}")
    return result


def _update_ssrd_metrics(
    frame: pd.DataFrame, ssrd: np.ndarray, target: np.ndarray,
    climatology: np.ndarray, mask: np.ndarray,
    by_lead: dict[tuple[str, int], MetricAccumulator],
    overall: dict[str, MetricAccumulator],
) -> None:
    for row in frame.itertuples(index=False):
        observation = np.asarray(target[int(row.dssr_target_index)], dtype=np.float32)
        history_indices = json.loads(row.history_dssr_indices)
        predictions = {
            "climatology": climatology[climatology_bin(date.fromisoformat(str(row.target_date_bjt)[:10]))],
            "persistence": np.asarray(target[int(history_indices[-1])], dtype=np.float32),
            "history_mean": np.asarray(target[history_indices], dtype=np.float32).mean(axis=0),
            "ssrd_ensemble_mean": ssrd[int(row.lead_day) - 1],
        }
        for name, prediction in predictions.items():
            by_lead[(name, int(row.lead_day))].update(prediction, observation, mask)
            overall[name].update(prediction, observation, mask)


def run(force: bool = False) -> tuple[list[Path], list[Path], list[str]]:
    paths = resolved_paths()
    stage04_done = PROJECT_ROOT / "state" / "stage_04_build_cache.done"
    train_path = PROJECT_ROOT / "manifests" / "samples_train.parquet"
    val_path = PROJECT_ROOT / "manifests" / "samples_val.parquet"
    test_path = PROJECT_ROOT / "manifests" / "samples_test.parquet"
    selected_path = PROJECT_ROOT / "manifests" / "selected_fields.parquet"
    target_grid_path = PROJECT_ROOT / "manifests" / "target_grid.json"
    required = [stage04_done, train_path, val_path, test_path, selected_path, target_grid_path]
    if not all(path.exists() for path in required):
        raise StageBlocked("Stage04 completion and frozen sample manifests are required")
    columns = [
        "sample_id", "init_date", "lead_day", "target_date_bjt",
        "dssr_target_index", "history_dssr_indices",
    ]
    train = pd.read_parquet(train_path, columns=columns)
    frames = {"validation": pd.read_parquet(val_path, columns=columns)}
    target_path = Path(paths["dssr"]["target"])
    mask_path = Path(paths["dssr"]["coverage_mask"])
    target = np.load(target_path, mmap_mode="r")
    mask = np.asarray(np.load(mask_path, mmap_mode="r"), dtype=bool)
    climatology, counts = _build_climatology(target, train)
    cache_path = Path(paths["work_root"]) / "cache" / "climatology_doy_train_float32.npy"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, climatology)
    metric_rows: list[dict[str, Any]] = []
    for split, frame in frames.items():
        accumulators = {(name, lead): MetricAccumulator() for name in ("climatology", "persistence", "history_mean")
                        for lead in range(1, 41)}
        overall = {name: MetricAccumulator() for name in ("climatology", "persistence", "history_mean")}
        for row in frame.itertuples(index=False):
            observation = np.asarray(target[int(row.dssr_target_index)], dtype=np.float32)
            history_indices = json.loads(row.history_dssr_indices)
            predictions = {
                "climatology": climatology[climatology_bin(date.fromisoformat(str(row.target_date_bjt)[:10]))],
                "persistence": np.asarray(target[int(history_indices[-1])], dtype=np.float32),
                "history_mean": np.asarray(target[history_indices], dtype=np.float32).mean(axis=0),
            }
            for name, prediction in predictions.items():
                accumulators[(name, int(row.lead_day))].update(prediction, observation, mask)
                overall[name].update(prediction, observation, mask)
        for (name, lead), accumulator in accumulators.items():
            metric_rows.append({"split": split, "cohort": "full40", "baseline": name,
                                "lead_day": lead, **accumulator.result()})
        for name, accumulator in overall.items():
            metric_rows.append({"split": split, "cohort": "full40", "baseline": name,
                                "lead_day": "overall", **accumulator.result()})
    target_grid = json.loads(target_grid_path.read_text(encoding="utf-8"))
    ssrd_files = _ssrd_file_map(selected_path)
    ssrd_counts: dict[str, dict[str, int]] = {}
    for split, frame in frames.items():
        matched = frame.loc[frame["init_date"].astype(str).str[:10].isin(ssrd_files)].copy()
        matched_inits = sorted(matched["init_date"].astype(str).str[:10].unique())
        if not matched_inits:
            raise StageBlocked(f"No SSRD-common init dates in {split}")
        ssrd_counts[split] = {"init_dates": len(matched_inits), "samples": len(matched)}
        names = ("climatology", "persistence", "history_mean", "ssrd_ensemble_mean")
        ssrd_acc = {(name, lead): MetricAccumulator() for name in names for lead in range(1, 41)}
        ssrd_all = {name: MetricAccumulator() for name in names}
        for position, (init_value, init_frame) in enumerate(matched.groupby("init_date"), start=1):
            init_text = str(init_value)[:10]
            ssrd = _load_ssrd_ensemble_mean(Path(ssrd_files[init_text]), target_grid)
            _update_ssrd_metrics(init_frame, ssrd, target, climatology, mask, ssrd_acc, ssrd_all)
            if position == 1 or position % 10 == 0 or position == len(matched_inits):
                print(f"Stage05 SSRD {split} {position}/{len(matched_inits)} init={init_text}", flush=True)
        for (name, lead), accumulator in ssrd_acc.items():
            metric_rows.append({
                "split": split, "cohort": "ssrd_common_init",
                "baseline": name, "lead_day": lead, **accumulator.result(),
            })
        for name, accumulator in ssrd_all.items():
            metric_rows.append({
                "split": split, "cohort": "ssrd_common_init",
                "baseline": name, "lead_day": "overall", **accumulator.result(),
            })

    metrics_path = PROJECT_ROOT / "outputs" / "metrics" / "baseline_metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
    report_path = PROJECT_ROOT / "outputs" / "reports" / "stage_05_baselines_summary.json"
    report_path.write_text(json.dumps({
        "climatology_basis": "unique training target dates only, 366-bin leap-day-preserving calendar",
        "climatology_bin_counts_min": int(counts.min()), "climatology_bin_counts_max": int(counts.max()),
        "evaluated_splits": {name: len(frame) for name, frame in frames.items()},
        "baselines": ["climatology", "persistence", "history_mean"],
        "test_policy": "not evaluated before Stage10 frozen-test lock",
        "ssrd_baseline": {
            "name": "ssrd_ensemble_mean",
            "source_units": "J m**-2 accumulated from forecast start",
            "conversion": "successive 24 h differences / 86400 s",
            "output_units": "W m**-2",
            "ensemble": "mean of members 0-10 after differencing",
            "spatial_mapping": "bilinear 8x15 to target 101x211 grid",
            "comparison_cohort": "same SSRD-available init dates for all baselines",
            "matched": ssrd_counts,
            "time_window_caveat": "SSRD is 00-24 UTC (08-08 BJT); DSSR is 06-20 BJT",
            "interpretation": "auxiliary approximate-window physical baseline only",
        },
    }, indent=2), encoding="utf-8")
    return [cache_path, metrics_path, report_path], required + [target_path, mask_path], []


__all__ = ["run", "climatology_bin"]
