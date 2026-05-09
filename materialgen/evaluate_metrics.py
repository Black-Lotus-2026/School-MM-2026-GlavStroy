"""Regression metrics (MAE, RMSE, MAPE, R²) for a trained NEAT-BNN model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import DatasetInputConfig, _resolve_config_path
from .data import prepare_dataset
from .neat_bnn import NeatBNNRegressor


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(y_true - y_pred), axis=0)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    eps = 1e-12
    denom = np.maximum(np.abs(y_true), eps)
    return np.mean(np.abs((y_true - y_pred) / denom), axis=0) * 100.0


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    y_mean = np.mean(y_true, axis=0)
    ss_tot = np.sum((y_true - y_mean) ** 2, axis=0)
    return np.where(ss_tot > 1e-15, 1.0 - ss_res / ss_tot, np.nan)


@dataclass
class EvaluateMetricsConfig:
    """JSON configuration for ``evaluate_metrics``."""

    model_path: str
    dataset: DatasetInputConfig
    validation_split: float = 0.2
    mc_samples: int = 30
    random_seed: int = 42

    @classmethod
    def from_dict(cls, payload: dict) -> "EvaluateMetricsConfig":
        ds_payload = payload.get("dataset", {})
        dataset = DatasetInputConfig.from_dict(
            ds_payload,
            legacy_data_path=payload.get("data_path"),
            legacy_components=payload.get("components"),
            legacy_properties=payload.get("properties"),
        )
        return cls(
            model_path=str(payload.get("model_path", "")),
            dataset=dataset,
            validation_split=float(payload.get("validation_split", 0.2)),
            mc_samples=int(payload.get("mc_samples", 30)),
            random_seed=int(payload.get("random_seed", 42)),
        )

    def resolve_paths(self, base_dir: Path) -> None:
        mp = _resolve_config_path(base_dir, self.model_path or None)
        self.model_path = "" if mp is None else mp
        self.dataset.resolve_paths(base_dir)

    def validate(self) -> None:
        if not self.model_path:
            raise ValueError("model_path is required")
        self.dataset.validate("dataset")
        if not (0.0 < self.validation_split < 1.0):
            raise ValueError("validation_split must be between 0 and 1")
        if self.mc_samples < 1:
            raise ValueError("mc_samples must be >= 1")


def load_evaluate_config(path: str | Path) -> EvaluateMetricsConfig:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    cfg = EvaluateMetricsConfig.from_dict(payload)
    cfg.resolve_paths(path.resolve().parent)
    cfg.validate()
    return cfg


def run_evaluate_metrics(config_path: str | Path) -> dict:
    """Evaluate MAE / RMSE / MAPE / R² on a held-out fraction of the dataset."""

    config = load_evaluate_config(config_path)

    regressor = NeatBNNRegressor.load(config.model_path)

    prepared = prepare_dataset(
        csv_path=config.dataset.data_path,
        component_columns=config.dataset.components,
        property_columns=config.dataset.properties,
    )

    n = len(prepared.components)
    rng = np.random.default_rng(config.random_seed)
    perm = rng.permutation(n)
    n_val = max(1, int(round(n * config.validation_split)))
    val_idx = perm[:n_val]

    props_val = prepared.properties[val_idx]
    comp_val = prepared.components[val_idx]

    pred_mean, pred_std = regressor.predict_components(props_val, mc_samples=config.mc_samples)

    out = {
        "model_path": str(Path(config.model_path).resolve()),
        "n_total": n,
        "n_validation": len(val_idx),
        "validation_split": config.validation_split,
        "mc_samples": config.mc_samples,
        "component_columns": list(config.dataset.components),
        "per_component": {},
        "aggregates": {},
    }

    mae_v = _mae(comp_val, pred_mean)
    rmse_v = _rmse(comp_val, pred_mean)
    mape_v = _mape(comp_val, pred_mean)
    r2_v = _r2_score(comp_val, pred_mean)

    for i, name in enumerate(config.dataset.components):
        out["per_component"][name] = {
            "MAE": float(mae_v[i]),
            "RMSE": float(rmse_v[i]),
            "MAPE": float(mape_v[i]),
            "R2": float(r2_v[i]),
            "pred_std_mean": float(np.mean(pred_std[:, i])),
        }

    out["aggregates"] = {
        "MAE_mean": float(np.mean(mae_v)),
        "RMSE_mean": float(np.mean(rmse_v)),
        "MAPE_mean": float(np.mean(mape_v)),
        "R2_mean": float(np.nanmean(r2_v)),
    }

    return out


__all__ = [
    "EvaluateMetricsConfig",
    "load_evaluate_config",
    "run_evaluate_metrics",
]
