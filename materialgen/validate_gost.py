"""Post-validation of forward predictions against GOST 26633 strength brands.

For each row in the dataset we:
  1) predict 28-day compressive strength with the trained forward BNN,
  2) look up the GOST class (Bxx) the prediction falls into using the
     [R_min, R_max] band defined by the GOST table,
  3) compare with the GOST class implied by the true measured strength.

The stage emits both per-row labels and aggregate metrics: exact match,
match within ±1 brand, and the share of cases where the true value sits
inside the model's predictive ±2σ interval.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import DatasetInputConfig, _resolve_config_path
from .data import _select_numeric_columns, read_dataset_frame
from .forward_model import ForwardBNNRegressor
from .stage_common import resolve_artifacts_layout, write_json

VALIDATE_GOST_STAGE_DIR = "validate_gost"


@dataclass
class GostBrand:
    label: str
    r_min: float
    r_max: float


def _load_gost_table(path: str | Path) -> list[GostBrand]:
    """Parse the GOST table CSV into a sorted list of strength brands."""

    frame = pd.read_csv(path, sep=";", skiprows=2, decimal=",", engine="python")
    label_col = next(c for c in frame.columns if c.startswith("Класс"))
    rmin_col = next(c for c in frame.columns if "min, МПа" in c)
    rmax_col = next(c for c in frame.columns if "max, МПа" in c)
    brands: list[GostBrand] = []
    for _, row in frame.iterrows():
        brands.append(
            GostBrand(
                label=str(row[label_col]).strip(),
                r_min=float(str(row[rmin_col]).replace(",", ".")),
                r_max=float(str(row[rmax_col]).replace(",", ".")),
            )
        )
    brands.sort(key=lambda b: b.r_min)
    return brands


def _classify(value: float, brands: list[GostBrand]) -> str:
    """Map a strength in MPa to the closest GOST brand label."""

    if value < brands[0].r_min:
        return f"<{brands[0].label}"
    if value > brands[-1].r_max:
        return f">{brands[-1].label}"
    for brand in brands:
        if brand.r_min <= value <= brand.r_max:
            return brand.label
    nearest = min(brands, key=lambda b: min(abs(b.r_min - value), abs(b.r_max - value)))
    return nearest.label


def _brand_index(label: str, brands: list[GostBrand]) -> int:
    """Position of the brand inside the sorted list, or sentinel for out-of-range."""

    for i, brand in enumerate(brands):
        if brand.label == label:
            return i
    if label.startswith("<"):
        return -1
    if label.startswith(">"):
        return len(brands)
    return -2


@dataclass
class ValidateGostConfig:
    model_path: str
    dataset: DatasetInputConfig
    gost_path: str
    mc_samples: int = 64
    seed: int = 42

    @classmethod
    def from_dict(cls, payload: dict) -> "ValidateGostConfig":
        ds = DatasetInputConfig.from_dict(payload.get("dataset", {}))
        return cls(
            model_path=str(payload["model_path"]),
            dataset=ds,
            gost_path=str(payload["gost_path"]),
            mc_samples=int(payload.get("mc_samples", 64)),
            seed=int(payload.get("seed", 42)),
        )

    def resolve_paths(self, base_dir: Path) -> None:
        self.dataset.resolve_paths(base_dir)
        self.model_path = _resolve_config_path(base_dir, self.model_path) or ""
        self.gost_path = _resolve_config_path(base_dir, self.gost_path) or ""

    def validate(self) -> None:
        if not self.model_path:
            raise ValueError("model_path is required")
        if not self.gost_path:
            raise ValueError("gost_path is required")
        self.dataset.validate("dataset")
        if self.mc_samples < 1:
            raise ValueError("mc_samples must be >= 1")


def load_validate_gost_config(path: str | Path) -> ValidateGostConfig:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    cfg = ValidateGostConfig.from_dict(payload)
    cfg.resolve_paths(path.resolve().parent)
    cfg.validate()
    return cfg


def run_validate_gost(
    *,
    config_path: str | Path,
    artifacts_dir: str | Path = "artifacts",
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Score a trained forward BNN against GOST strength brands."""

    config = load_validate_gost_config(config_path)

    layout = resolve_artifacts_layout(artifacts_dir, inverse_dir=None, bnn_dir=None)
    target_dir = Path(out_dir) if out_dir else layout.root / VALIDATE_GOST_STAGE_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    regressor = ForwardBNNRegressor.load(config.model_path)

    frame = read_dataset_frame(config.dataset.data_path, skiprows=config.dataset.skiprows)
    components = _select_numeric_columns(frame, config.dataset.components, config.dataset.data_path)
    target_frame = _select_numeric_columns(frame, config.dataset.properties, config.dataset.data_path)
    if target_frame.shape[1] != 1:
        raise ValueError("validate_gost expects exactly one strength target column")
    x = components.to_numpy(dtype=np.float32)
    y_true = target_frame.to_numpy(dtype=np.float32).ravel()

    pred_mean, pred_std = regressor.predict(x, mc_samples=config.mc_samples)
    pred_mean = pred_mean.ravel()
    pred_std = pred_std.ravel()

    brands = _load_gost_table(config.gost_path)
    pred_class = [_classify(float(v), brands) for v in pred_mean]
    true_class = [_classify(float(v), brands) for v in y_true]

    pred_idx = np.array([_brand_index(c, brands) for c in pred_class])
    true_idx = np.array([_brand_index(c, brands) for c in true_class])
    in_range_mask = (pred_idx >= 0) & (pred_idx < len(brands)) & (true_idx >= 0) & (true_idx < len(brands))

    exact = float(np.mean(pred_idx == true_idx))
    within_one = float(np.mean(np.abs(pred_idx - true_idx) <= 1))
    within_one_in_range = (
        float(np.mean(np.abs(pred_idx[in_range_mask] - true_idx[in_range_mask]) <= 1))
        if in_range_mask.any() else 0.0
    )

    lower = pred_mean - 2.0 * pred_std
    upper = pred_mean + 2.0 * pred_std
    coverage_2sigma = float(np.mean((y_true >= lower) & (y_true <= upper)))

    matrix = pd.crosstab(
        pd.Series(true_class, name="true"),
        pd.Series(pred_class, name="pred"),
        dropna=False,
    )

    rows = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred_mean": pred_mean,
            "y_pred_std": pred_std,
            "true_brand": true_class,
            "pred_brand": pred_class,
        }
    )
    rows.to_csv(target_dir / "predictions.csv", index=False)
    matrix.to_csv(target_dir / "confusion.csv")

    summary = {
        "stage": "validate_gost",
        "model_path": str(config.model_path),
        "data_path": str(config.dataset.data_path),
        "gost_path": str(config.gost_path),
        "n_total": int(len(y_true)),
        "n_in_gost_range": int(in_range_mask.sum()),
        "metrics": {
            "exact_match": exact,
            "within_one_class": within_one,
            "within_one_class_in_range_only": within_one_in_range,
            "predictive_2sigma_coverage": coverage_2sigma,
        },
        "predictions_path": str(target_dir / "predictions.csv"),
        "confusion_path": str(target_dir / "confusion.csv"),
    }
    write_json(target_dir / "validation_summary.json", summary)
    return summary


__all__ = [
    "ValidateGostConfig",
    "load_validate_gost_config",
    "run_validate_gost",
]
