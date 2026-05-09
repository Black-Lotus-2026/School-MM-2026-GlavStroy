"""Forward stage: train ``composition → properties`` Bayesian MLP and report
MAE / RMSE / MAPE / R² on a held-out fold.

This is a self-contained pipeline: it does not depend on the inverse NEAT
artifacts. ``components`` in the config = inputs to the model, ``properties``
= multi-output regression targets.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import DatasetInputConfig, _resolve_config_path
from .data import prepare_dataset
from .forward_model import ForwardBNNRegressor
from .stage_common import resolve_artifacts_layout, write_json

FORWARD_STAGE_DIR = "train_forward"


@dataclass
class ForwardStageConfig:
    """JSON configuration for the forward training stage."""

    dataset: DatasetInputConfig
    hidden_layers: list[int] = field(default_factory=lambda: [64, 32])
    prior_std: float = 1.0
    learning_rate: float = 0.005
    epochs: int = 200
    batch_size: int = 32
    validation_split: float = 0.2
    early_stopping_rounds: int = 30
    mc_samples: int = 30
    eval_mc_samples: int = 50
    seed: int = 42
    pretrained_model_path: str | None = None

    @classmethod
    def from_dict(cls, payload: dict) -> "ForwardStageConfig":
        ds = DatasetInputConfig.from_dict(payload.get("dataset", {}))
        return cls(
            dataset=ds,
            hidden_layers=list(payload.get("hidden_layers", [64, 32])),
            prior_std=float(payload.get("prior_std", 1.0)),
            learning_rate=float(payload.get("learning_rate", 0.005)),
            epochs=int(payload.get("epochs", 200)),
            batch_size=int(payload.get("batch_size", 32)),
            validation_split=float(payload.get("validation_split", 0.2)),
            early_stopping_rounds=int(payload.get("early_stopping_rounds", 30)),
            mc_samples=int(payload.get("mc_samples", 30)),
            eval_mc_samples=int(payload.get("eval_mc_samples", 50)),
            seed=int(payload.get("seed", 42)),
            pretrained_model_path=payload.get("pretrained_model_path"),
        )

    def resolve_paths(self, base_dir: Path) -> None:
        self.dataset.resolve_paths(base_dir)
        if self.pretrained_model_path:
            resolved = _resolve_config_path(base_dir, self.pretrained_model_path)
            self.pretrained_model_path = resolved

    def validate(self) -> None:
        self.dataset.validate("dataset")
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if not (0.0 < self.validation_split < 1.0):
            raise ValueError("validation_split must be in (0, 1)")
        if self.mc_samples < 1 or self.eval_mc_samples < 1:
            raise ValueError("mc_samples must be >= 1")

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset.to_dict(),
            "hidden_layers": self.hidden_layers,
            "prior_std": self.prior_std,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "validation_split": self.validation_split,
            "early_stopping_rounds": self.early_stopping_rounds,
            "mc_samples": self.mc_samples,
            "eval_mc_samples": self.eval_mc_samples,
            "seed": self.seed,
            "pretrained_model_path": self.pretrained_model_path,
        }


def load_forward_config(path: str | Path) -> ForwardStageConfig:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    cfg = ForwardStageConfig.from_dict(payload)
    cfg.resolve_paths(path.resolve().parent)
    cfg.validate()
    return cfg


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, np.ndarray]:
    eps = 1e-12
    mae = np.mean(np.abs(y_true - y_pred), axis=0)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps)), axis=0) * 100.0
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    y_mean = np.mean(y_true, axis=0)
    ss_tot = np.sum((y_true - y_mean) ** 2, axis=0)
    r2 = np.where(ss_tot > 1e-15, 1.0 - ss_res / ss_tot, np.nan)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}


def run_train_forward(
    *,
    config_path: str | Path,
    artifacts_dir: str | Path = "artifacts",
    forward_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Train a forward Bayesian MLP and store metrics + checkpoint."""

    config = load_forward_config(config_path)

    layout = resolve_artifacts_layout(artifacts_dir, inverse_dir=None, bnn_dir=None)
    out_dir = Path(forward_dir) if forward_dir else layout.root / FORWARD_STAGE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = prepare_dataset(
        csv_path=config.dataset.data_path,
        component_columns=config.dataset.components,
        property_columns=config.dataset.properties,
    )

    x = np.asarray(dataset.components, dtype=np.float32)
    y = np.asarray(dataset.properties, dtype=np.float32)

    rng = np.random.default_rng(config.seed)
    perm = rng.permutation(len(x))
    n_val = max(1, int(round(len(x) * config.validation_split)))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:] if n_val < len(x) else perm

    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]

    if config.pretrained_model_path and Path(config.pretrained_model_path).exists():
        regressor = ForwardBNNRegressor.load(config.pretrained_model_path)
        if regressor.input_dim != x.shape[1] or regressor.output_dim != y.shape[1]:
            raise ValueError(
                "Pretrained model dimensions do not match the dataset: "
                f"got input={regressor.input_dim}, output={regressor.output_dim}; "
                f"expected input={x.shape[1]}, output={y.shape[1]}"
            )
    else:
        regressor = ForwardBNNRegressor(
            input_dim=x.shape[1],
            output_dim=y.shape[1],
            hidden_layers=list(config.hidden_layers),
            prior_std=config.prior_std,
            seed=config.seed,
        )

    training_result = regressor.fit(
        x_train,
        y_train,
        learning_rate=config.learning_rate,
        epochs=config.epochs,
        batch_size=config.batch_size,
        validation_split=config.validation_split,
        early_stopping_rounds=config.early_stopping_rounds,
        mc_samples=config.mc_samples,
    )

    pred_val_mean, pred_val_std = regressor.predict(x_val, mc_samples=config.eval_mc_samples)
    metrics_val = _regression_metrics(y_val, pred_val_mean)
    pred_train_mean, _ = regressor.predict(x_train, mc_samples=config.eval_mc_samples)
    metrics_train = _regression_metrics(y_train, pred_train_mean)

    model_path = out_dir / "forward_bnn.pt"
    regressor.save(model_path)
    write_json(out_dir / "forward_config.json", config.to_dict())

    summary: dict[str, Any] = {
        "stage": "train_forward",
        "data_path": str(config.dataset.data_path),
        "n_total": int(len(x)),
        "n_train": int(len(train_idx)),
        "n_validation": int(len(val_idx)),
        "input_columns": list(config.dataset.components),
        "output_columns": list(config.dataset.properties),
        "training": training_result,
        "metrics_validation": {
            "per_target": {
                name: {
                    "MAE": float(metrics_val["MAE"][i]),
                    "RMSE": float(metrics_val["RMSE"][i]),
                    "MAPE": float(metrics_val["MAPE"][i]),
                    "R2": float(metrics_val["R2"][i]),
                    "pred_std_mean": float(np.mean(pred_val_std[:, i])),
                }
                for i, name in enumerate(config.dataset.properties)
            },
            "aggregates": {
                "MAE_mean": float(np.mean(metrics_val["MAE"])),
                "RMSE_mean": float(np.mean(metrics_val["RMSE"])),
                "MAPE_mean": float(np.mean(metrics_val["MAPE"])),
                "R2_mean": float(np.nanmean(metrics_val["R2"])),
            },
        },
        "metrics_train": {
            "aggregates": {
                "MAE_mean": float(np.mean(metrics_train["MAE"])),
                "RMSE_mean": float(np.mean(metrics_train["RMSE"])),
                "MAPE_mean": float(np.mean(metrics_train["MAPE"])),
                "R2_mean": float(np.nanmean(metrics_train["R2"])),
            },
        },
        "model_path": str(model_path),
    }
    write_json(out_dir / "training_summary.json", summary)
    return summary


__all__ = [
    "ForwardStageConfig",
    "load_forward_config",
    "run_train_forward",
]
