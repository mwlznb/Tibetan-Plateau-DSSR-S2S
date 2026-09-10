"""Stage 10 first and only frozen evaluation on 2022-2023 targets."""
from __future__ import annotations

import json
import math
import os
import pickle
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
import zarr
from torch.utils.data import DataLoader

from src.baselines.run import MetricAccumulator, climatology_bin
from src.diffusion.common import build_center_cache, forecast_dataset
from src.diffusion.sampling import build_ensemble_cache
from src.evaluation.deterministic import evaluate
from src.models.deterministic import S2SDSSRUnet
from src.training.deterministic import ForecastDataset
from src.utils.config import PROJECT_ROOT, load_yaml, resolved_paths


class ProbabilityAccumulator:
    def __init__(self) -> None:
        self.points = 0; self.crps = self.covered = self.width = self.spread = 0.0
        self.mean_metrics = MetricAccumulator()

    def update(self, members: np.ndarray, observation: np.ndarray, mask: np.ndarray) -> None:
        values = np.asarray(members[:, mask], dtype=np.float64)
        truth = np.asarray(observation[mask], dtype=np.float64)
        count = values.shape[0]
        first = np.abs(values - truth[None]).mean(axis=0)
        ordered = np.sort(values, axis=0)
        coefficient = 2 * np.arange(1, count + 1) - count - 1
        second = (ordered * coefficient[:, None]).sum(axis=0) / (count * count)
        lower, upper = np.quantile(values, [0.1, 0.9], axis=0)
        self.points += truth.size; self.crps += (first - second).sum()
        self.covered += ((truth >= lower) & (truth <= upper)).sum()
        self.width += (upper - lower).sum(); self.spread += values.std(axis=0, ddof=1).sum()
        self.mean_metrics.update(values.mean(axis=0), truth, np.ones(truth.shape, dtype=bool))

    def result(self, deterministic_crps: float) -> dict[str, Any]:
        center = self.mean_metrics.result(); crps = self.crps / self.points
        return {"points": self.points, "empirical_crps": crps,
                "crps_skill_score": 1.0 - crps / deterministic_crps,
                "p10_p90_coverage": self.covered / self.points,
                "p10_p90_width_wm2": self.width / self.points,
                "spread_wm2": self.spread / self.points,
                "spread_rmse_ratio": (self.spread / self.points) / center["rmse"],
                "ensemble_mean_rmse_wm2": center["rmse"], "ensemble_mean_mae_wm2": center["mae"],
                "ensemble_mean_pcc": center["pcc"], "ensemble_mean_bias_wm2": center["bias"]}


def _deterministic_baselines(dataset: ForecastDataset, center_path: Path) -> pd.DataFrame:
    center = zarr.open_group(str(center_path), mode="r")["data"]
    names = ("climatology", "persistence", "history_mean", "D2_main")
    metrics = {(name, lead): MetricAccumulator() for name in names for lead in range(1, 41)}
    anomaly = {(name, lead): MetricAccumulator() for name in names for lead in range(1, 41)}
    overall = {name: MetricAccumulator() for name in names}
    anomaly_all = {name: MetricAccumulator() for name in names}
    mask = dataset.mask
    for index in range(len(dataset)):
        sample = dataset[index]; lead = int(sample["lead"])
        observation = sample["target"].numpy()[0] * dataset.dssr_scale
        climatology = sample["climatology"].numpy()[0] * dataset.dssr_scale
        history = sample["context"].numpy()[:7] * dataset.dssr_scale
        predictions = {"climatology": climatology, "persistence": history[-1],
                       "history_mean": history.mean(axis=0),
                       "D2_main": np.asarray(center[index], dtype=np.float32)}
        for name, prediction in predictions.items():
            metrics[(name, lead)].update(prediction, observation, mask)
            overall[name].update(prediction, observation, mask)
            anomaly[(name, lead)].update(prediction - climatology, observation - climatology, mask)
            anomaly_all[name].update(prediction - climatology, observation - climatology, mask)
        if index == 0 or (index + 1) % 250 == 0 or index + 1 == len(dataset):
            print(f"Stage10 deterministic baselines {index + 1}/{len(dataset)}", flush=True)
    rows = []
    for lead_value in list(range(1, 41)) + ["overall"]:
        clim = (overall["climatology"].result() if lead_value == "overall" else
                metrics[("climatology", lead_value)].result())
        for name in names:
            result = (overall[name].result() if lead_value == "overall" else
                      metrics[(name, lead_value)].result())
            acc = (anomaly_all[name].result()["pcc"] if lead_value == "overall" else
                   anomaly[(name, lead_value)].result()["pcc"])
            rows.append({"model": name, "lead_day": lead_value, **result,
                         "rmse_skill_score": 1.0 - result["rmse"] / clim["rmse"],
                         "mae_skill_score": 1.0 - result["mae"] / clim["mae"],
                         "anomaly_correlation": acc})
    return pd.DataFrame(rows)


def _ablation_metrics(name: str, device: torch.device, climatology: np.ndarray,
                      reference: pd.DataFrame) -> pd.DataFrame | None:
    root = PROJECT_ROOT / "outputs" / "stage07_ablations" / name
    summary_path, config_path = root / "run_summary.json", root / "config.yaml"
    if not summary_path.exists() or not config_path.exists(): return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "PASS": return None
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")); paths = resolved_paths()
    normalization_name = "s2s_control_normalization.json" if cfg["s2s_mode"] == "control" else "s2s_normalization.json"
    normalization = json.loads((PROJECT_ROOT / "manifests" / normalization_name).read_text(encoding="utf-8"))
    schema = json.loads((PROJECT_ROOT / "manifests" / "feature_schema.json").read_text(encoding="utf-8"))
    index_frame = pd.read_parquet(PROJECT_ROOT / "manifests" / "cache_init_dates.parquet")
    indices = dict(zip(index_frame.init_date.astype(str), index_frame.cache_init_index.astype(int)))
    channel_indices = ([idx for idx, value in enumerate(schema["channels"]) if not value.endswith("_ensstd")]
                       if cfg["s2s_mode"] == "ensemble_mean" else None)
    dataset = ForecastDataset(PROJECT_ROOT / "manifests" / "samples_test.parquet", indices, paths,
                              normalization, float(load_yaml("data.yaml")["dssr_scale_wm2"]),
                              no_history=bool(cfg.get("no_history", False)), climatology=climatology,
                              target_type=cfg["target_type"],
                              s2s_cache_name="s2s_control_features.zarr" if cfg["s2s_mode"] == "control" else "s2s_features.zarr",
                              s2s_channel_indices=channel_indices)
    checkpoint = torch.load(Path(summary["checkpoint"]), map_location=device, weights_only=False)
    model = S2SDSSRUnet(int(checkpoint["C_s2s"]), int(cfg["lead_embedding_dim"]),
                        int(cfg["base_channels"]), float(cfg["dropout"]), cfg["target_type"]).to(device)
    model.load_state_dict(checkpoint["model"])
    loader = DataLoader(dataset, batch_size=int(checkpoint["batch_size"]), shuffle=False,
                        num_workers=0, pin_memory=True)
    mask = torch.from_numpy(dataset.mask.astype(np.float32))[None, None].to(device)
    result = evaluate(model, loader, device, mask, bool(cfg["amp"]), dataset.dssr_scale, climatology)
    result["model"] = name
    result["cohort"] = "full40"
    result["comparability_note"] = "same test cohort and DSSR target window"
    result = result.rename(columns={"rmse_wm2": "rmse", "mae_wm2": "mae", "bias_wm2": "bias"})
    climatology_rows = reference.loc[reference.model == "climatology"].copy()
    climatology_rows.index = climatology_rows.lead_day.astype(str)
    result["mae_skill_score"] = [
        1.0 - row.mae / climatology_rows.loc[str(row.lead_day), "mae"] for row in result.itertuples()
    ]
    return result


def _probability_metrics(dataset: ForecastDataset, center_path: Path, ensemble_path: Path,
                         s_res: float, s_rec: float) -> pd.DataFrame:
    center = zarr.open_group(str(center_path), mode="r")["data"]
    residual = zarr.open_group(str(ensemble_path), mode="r")["residual_members"]
    methods = ("deterministic_degenerate", "residual_diffusion", "recentered_residual_diffusion")
    by_lead = {(name, lead): ProbabilityAccumulator() for name in methods for lead in range(1, 41)}
    overall = {name: ProbabilityAccumulator() for name in methods}
    checkpoint_path = (PROJECT_ROOT / "outputs" / "metrics" /
                       "stage10_probability_metrics_checkpoint.pkl")
    signature = {"samples": len(dataset), "center_path": str(center_path),
                 "ensemble_path": str(ensemble_path), "s_res": float(s_res),
                 "s_rec": float(s_rec), "methods": methods}
    start_index = 0
    if checkpoint_path.exists():
        with checkpoint_path.open("rb") as stream:
            state = pickle.load(stream)
        if state.get("signature") != signature:
            raise RuntimeError("Incompatible Stage10 probability-metrics checkpoint")
        start_index = int(state["next_index"])
        by_lead, overall = state["by_lead"], state["overall"]
        print(f"Stage10 probability metrics resume {start_index}/{len(dataset)}", flush=True)
    for index in range(start_index, len(dataset)):
        sample = dataset[index]; lead = int(sample["lead"])
        observation = sample["target"].numpy()[0] * dataset.dssr_scale
        center_value = np.asarray(center[index], dtype=np.float32)
        draws = np.asarray(residual[index], dtype=np.float32)
        centered = draws - draws.mean(axis=0, keepdims=True)
        ensembles = {
            "deterministic_degenerate": np.repeat(center_value[None], draws.shape[0], axis=0),
            "residual_diffusion": np.maximum(center_value[None] + s_res * draws, 0.0),
            "recentered_residual_diffusion": np.maximum(center_value[None] + s_rec * centered, 0.0),
        }
        for name, members in ensembles.items():
            by_lead[(name, lead)].update(members, observation, dataset.mask)
            overall[name].update(members, observation, dataset.mask)
        if index == 0 or (index + 1) % 50 == 0 or index + 1 == len(dataset):
            print(f"Stage10 probability metrics {index + 1}/{len(dataset)}", flush=True)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(str(checkpoint_path) + ".working")
            with temporary.open("wb") as stream:
                pickle.dump({"signature": signature, "next_index": index + 1,
                             "by_lead": by_lead, "overall": overall},
                            stream, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temporary, checkpoint_path)
    rows = []
    for lead_value in list(range(1, 41)) + ["overall"]:
        deterministic = (overall["deterministic_degenerate"] if lead_value == "overall" else
                         by_lead[("deterministic_degenerate", lead_value)])
        deterministic_crps = deterministic.crps / deterministic.points
        for name in methods:
            accumulator = overall[name] if lead_value == "overall" else by_lead[(name, lead_value)]
            rows.append({"method": name, "lead_day": lead_value,
                         **accumulator.result(deterministic_crps)})
    return pd.DataFrame(rows)


def _ssrd_auxiliary_test(center_path: Path) -> pd.DataFrame:
    from src.baselines.run import _load_ssrd_ensemble_mean, _ssrd_file_map
    paths = resolved_paths()
    columns = ["init_date", "lead_day", "target_date_bjt", "dssr_target_index"]
    frame = pd.read_parquet(PROJECT_ROOT / "manifests" / "samples_test.parquet", columns=columns)
    frame["sample_position"] = np.arange(len(frame), dtype=np.int64)
    files = _ssrd_file_map(PROJECT_ROOT / "manifests" / "selected_fields.parquet")
    matched = frame.loc[frame.init_date.astype(str).str[:10].isin(files)].copy()
    if matched.empty:
        raise RuntimeError("No SSRD-common init dates in frozen test cohort")
    per_init = matched.groupby(matched.init_date.astype(str).str[:10]).lead_day.nunique()
    if not bool((per_init == 40).all()):
        raise RuntimeError("SSRD-common frozen cohort is not full40")
    target = np.load(Path(paths["dssr"]["target"]), mmap_mode="r")
    mask = np.asarray(np.load(Path(paths["dssr"]["coverage_mask"]), mmap_mode="r"), dtype=bool)
    climatology = np.load(Path(paths["work_root"]) / "cache" /
                          "climatology_doy_train_float32.npy", mmap_mode="r")
    grid = json.loads((PROJECT_ROOT / "manifests" / "target_grid.json").read_text(encoding="utf-8"))
    ssrd = {lead: MetricAccumulator() for lead in range(1, 41)}
    d2 = {lead: MetricAccumulator() for lead in range(1, 41)}
    clim = {lead: MetricAccumulator() for lead in range(1, 41)}
    anomaly = {lead: MetricAccumulator() for lead in range(1, 41)}
    d2_anomaly = {lead: MetricAccumulator() for lead in range(1, 41)}
    ssrd_all, d2_all, clim_all = MetricAccumulator(), MetricAccumulator(), MetricAccumulator()
    anomaly_all, d2_anomaly_all = MetricAccumulator(), MetricAccumulator()
    center = zarr.open_group(str(center_path), mode="r")["data"]
    for number, (init_value, group) in enumerate(matched.groupby("init_date"), start=1):
        init_text = str(init_value)[:10]
        fields = _load_ssrd_ensemble_mean(Path(files[init_text]), grid)
        for row in group.itertuples(index=False):
            lead = int(row.lead_day)
            observation = np.asarray(target[int(row.dssr_target_index)], dtype=np.float32)
            climate = climatology[climatology_bin(date.fromisoformat(str(row.target_date_bjt)[:10]))]
            prediction = fields[lead - 1]
            d2_prediction = np.asarray(center[int(row.sample_position)], dtype=np.float32)[::-1]
            ssrd[lead].update(prediction, observation, mask); ssrd_all.update(prediction, observation, mask)
            d2[lead].update(d2_prediction, observation, mask); d2_all.update(d2_prediction, observation, mask)
            clim[lead].update(climate, observation, mask); clim_all.update(climate, observation, mask)
            anomaly[lead].update(prediction - climate, observation - climate, mask)
            anomaly_all.update(prediction - climate, observation - climate, mask)
            d2_anomaly[lead].update(d2_prediction - climate, observation - climate, mask)
            d2_anomaly_all.update(d2_prediction - climate, observation - climate, mask)
        if number == 1 or number % 10 == 0:
            print(f"Stage10 SSRD auxiliary {number} init dates", flush=True)
    rows = []
    for lead_value in list(range(1, 41)) + ["overall"]:
        reference = clim_all.result() if lead_value == "overall" else clim[lead_value].result()
        for name, accumulator, anomaly_accumulator in (
                ("model_forecast_D2_ssrd_common", d2_all if lead_value == "overall" else d2[lead_value],
                 d2_anomaly_all if lead_value == "overall" else d2_anomaly[lead_value]),
                ("ssrd_ensemble_mean", ssrd_all if lead_value == "overall" else ssrd[lead_value],
                 anomaly_all if lead_value == "overall" else anomaly[lead_value])):
            result = accumulator.result(); acc = anomaly_accumulator.result()["pcc"]
            rows.append({"model": name, "lead_day": lead_value, **result,
                         "rmse_skill_score": 1.0 - result["rmse"] / reference["rmse"],
                         "mae_skill_score": 1.0 - result["mae"] / reference["mae"],
                         "anomaly_correlation": acc, "cohort": "ssrd_common_init",
                         "init_dates": int(len(per_init)), "samples": int(len(matched)),
                         "verification_target": "DSSR_observed_truth",
                         "forecast_source": "trained_U-Net" if name.startswith("model_forecast") else "ECMWF_SSRD",
                         "comparability_note": "auxiliary: same init/sample cohort; SSRD 00-24 UTC versus DSSR target 06-20 BJT"})
    return pd.DataFrame(rows)


def run() -> None:
    done = PROJECT_ROOT / "state" / "stage_10_frozen_test.done"
    if done.exists(): print("Stage10 already complete; skipping", flush=True); return
    cfg = load_yaml("evaluation.yaml")
    lock_path = PROJECT_ROOT / "state" / "FROZEN_TEST.lock"
    if not lock_path.exists(): raise RuntimeError("FROZEN_TEST.lock is required")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    scale = float(cfg["residual_scale"])
    from src.utils.hashing import sha256_file
    for name, record in lock["files"].items():
        if sha256_file(Path(record["path"])) != record["sha256"]:
            raise RuntimeError(f"Frozen file changed after lock: {name}")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required for Stage10")
    device = torch.device("cuda")
    center_path = build_center_cache("test", device)
    ensemble_path = build_ensemble_cache("test", device)
    dataset = forecast_dataset("test")
    deterministic = _deterministic_baselines(dataset, center_path)
    deterministic["cohort"] = "full40"
    deterministic["verification_target"] = "DSSR_observed_truth"
    deterministic["forecast_source"] = deterministic.model.map(
        {"D2_main": "trained_U-Net", "climatology": "DSSR_train_climatology",
         "persistence": "pre-init_DSSR", "history_mean": "pre-init_DSSR"}).fillna("trained_U-Net")
    deterministic["comparability_note"] = "same test cohort and DSSR target window"
    climatology = np.load(Path(resolved_paths()["work_root"]) / "cache" /
                          "climatology_doy_train_float32.npy", mmap_mode="r")
    for name in ("D0_control_only", "D1_ensemble_mean"):
        result = _ablation_metrics(name, device, climatology, deterministic)
        if result is not None:
            deterministic = pd.concat((deterministic, result), ignore_index=True, sort=False)
    try:
        deterministic = pd.concat((deterministic, _ssrd_auxiliary_test(center_path)), ignore_index=True, sort=False)
        ssrd_status = "PASS_AUXILIARY_DIFFERENT_TIME_WINDOW"
    except Exception as exc:
        ssrd_status = f"NOT EXECUTED — {exc!r}"
        print(f"Stage10 SSRD auxiliary failed: {exc!r}; continuing", flush=True)
    deterministic_path = PROJECT_ROOT / "outputs" / "metrics" / "test_deterministic_by_lead.csv"
    deterministic.to_csv(deterministic_path, index=False)
    common_overall = deterministic.loc[
        deterministic.model.isin(("model_forecast_D2_ssrd_common", "ssrd_ensemble_mean")) &
        (deterministic.lead_day.astype(str) == "overall")].copy()
    comparison_path = PROJECT_ROOT / "outputs" / "tables" / "stage10_model_vs_ssrd_common.csv"
    common_overall.to_csv(comparison_path, index=False)
    if len(common_overall) == 2:
        ordered = common_overall.sort_values("rmse")
        winner = str(ordered.iloc[0].model)
        rmse_difference = float(ordered.iloc[1].rmse - ordered.iloc[0].rmse)
        comparison_summary = {
            "status": "PASS", "verification_target": "DSSR_observed_truth",
            "same_init_lead_spatial_cohort": True, "winner_by_rmse": winner,
            "rmse_advantage_wm2": rmse_difference,
            "time_window_caveat": "SSRD 00-24 UTC; DSSR truth and model target 06-20 BJT",
            "table": str(comparison_path),
        }
    else:
        comparison_summary = {"status": "NOT_EXECUTED", "reason": ssrd_status,
                              "verification_target": "DSSR_observed_truth",
                              "table": str(comparison_path)}
    probability = _probability_metrics(dataset, center_path, ensemble_path,
                                       scale, scale)
    probability_path = PROJECT_ROOT / "outputs" / "metrics" / "test_probabilistic_by_lead.csv"
    probability.to_csv(probability_path, index=False)
    det_overall = deterministic.loc[(deterministic.model == "D2_main") &
                                    (deterministic.lead_day.astype(str) == "overall")].iloc[0]
    prob_overall = probability.loc[probability.lead_day.astype(str) == "overall"].set_index("method")
    report = {"stage": 10, "status": "done", "frozen_lock_sha256": sha256_file(lock_path),
              "deterministic_test": {"rmse_wm2": float(det_overall.rmse),
                                     "mae_wm2": float(det_overall.mae), "pcc": float(det_overall.pcc),
                                     "bias_wm2": float(det_overall.bias),
                                     "rmse_skill_score": float(det_overall.rmse_skill_score),
                                     "mae_skill_score": float(det_overall.mae_skill_score),
                                     "anomaly_correlation": float(det_overall.anomaly_correlation)},
              "probabilistic_test": prob_overall.to_dict(orient="index"),
              "s_res": scale, "s_rec": scale,
              "ssrd_auxiliary": ssrd_status,
              "model_forecast_vs_ssrd": comparison_summary,
              "post_test_retuning_policy": "forbidden"}
    report_path = PROJECT_ROOT / "outputs" / "reports" / "stage_10_frozen_test_summary.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    done.write_text(json.dumps(report, indent=2), encoding="utf-8")


__all__ = ["run", "ProbabilityAccumulator"]
