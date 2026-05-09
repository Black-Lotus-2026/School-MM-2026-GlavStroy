"""Forward Bayesian MLP: ``mix composition → properties``.

Independent of the inverse NEAT pipeline. Designed for the well-determined
regression that the case actually asks for (predict strength / slump from a
given composition). All weights are Bayesian (Normal prior + AutoNormal
variational posterior) so we keep epistemic uncertainty as required by the
case description.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
from pyro.infer import SVI, Predictive, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.nn import PyroModule, PyroSample

from .scaler import StandardScaler


class ForwardBayesianMLP(PyroModule):
    """Bayesian MLP with Gaussian prior on every weight and bias."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: list[int],
        prior_std: float = 1.0,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_layers = list(hidden_layers)
        self.prior_std = float(prior_std)

        modules: list[nn.Module] = []
        prev = self.input_dim
        for h in self.hidden_layers:
            linear = PyroModule[nn.Linear](prev, h)
            linear.weight = PyroSample(
                dist.Normal(0.0, self.prior_std).expand([h, prev]).to_event(2)
            )
            linear.bias = PyroSample(
                dist.Normal(0.0, self.prior_std).expand([h]).to_event(1)
            )
            modules.append(linear)
            modules.append(nn.ReLU())
            prev = h

        head = PyroModule[nn.Linear](prev, self.output_dim)
        head.weight = PyroSample(
            dist.Normal(0.0, self.prior_std)
            .expand([self.output_dim, prev]).to_event(2)
        )
        head.bias = PyroSample(
            dist.Normal(0.0, self.prior_std).expand([self.output_dim]).to_event(1)
        )
        modules.append(head)
        self.network = PyroModule[nn.Sequential](*modules)

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        sigma = pyro.sample(
            "sigma",
            dist.HalfNormal(1.0).expand([self.output_dim]).to_event(1),
        )
        mean = self.network(x)
        with pyro.plate("data", x.shape[0]):
            pyro.sample(
                "obs",
                dist.Normal(mean, sigma).to_event(1),
                obs=y,
            )
        return mean


@dataclass
class ForwardBNNRegressor:
    """High-level wrapper: scale → fit (SVI) → MC predict → save/load."""

    input_dim: int
    output_dim: int
    hidden_layers: list[int]
    prior_std: float = 1.0
    seed: int = 42

    def __post_init__(self) -> None:
        torch.manual_seed(self.seed)
        self.model = ForwardBayesianMLP(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_layers=self.hidden_layers,
            prior_std=self.prior_std,
        )
        self.guide = AutoNormal(self.model)
        self.x_scaler: StandardScaler | None = None
        self.y_scaler: StandardScaler | None = None

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        learning_rate: float = 1e-2,
        epochs: int = 200,
        batch_size: int = 32,
        validation_split: float = 0.2,
        early_stopping_rounds: int = 30,
        mc_samples: int = 30,
    ) -> dict[str, Any]:
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 1:
            y = y[:, None]
        assert x.shape[1] == self.input_dim, (
            f"Input dim mismatch: expected {self.input_dim}, got {x.shape[1]}"
        )
        assert y.shape[1] == self.output_dim, (
            f"Output dim mismatch: expected {self.output_dim}, got {y.shape[1]}"
        )

        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(len(x))
        n_val = max(1, int(round(len(x) * validation_split)))
        val_idx = idx[:n_val]
        train_idx = idx[n_val:] if n_val < len(x) else idx

        self.x_scaler = StandardScaler.fit(x[train_idx])
        self.y_scaler = StandardScaler.fit(y[train_idx])

        x_train = self.x_scaler.transform(x[train_idx]).astype(np.float32)
        y_train = self.y_scaler.transform(y[train_idx]).astype(np.float32)
        x_val = self.x_scaler.transform(x[val_idx]).astype(np.float32)
        y_val = self.y_scaler.transform(y[val_idx]).astype(np.float32)

        pyro.clear_param_store()
        optim = pyro.optim.Adam({"lr": float(learning_rate)})
        svi = SVI(self.model, self.guide, optim, loss=Trace_ELBO())

        best_val = float("inf")
        best_state: Any = None
        epochs_no_improve = 0
        train_losses: list[float] = []
        val_losses: list[float] = []

        for epoch in range(int(epochs)):
            perm = rng.permutation(len(x_train))
            x_e = x_train[perm]
            y_e = y_train[perm]

            for start in range(0, len(x_e), batch_size):
                stop = start + batch_size
                xb = torch.from_numpy(x_e[start:stop])
                yb = torch.from_numpy(y_e[start:stop])
                svi.step(xb, yb)

            train_pred, _ = self._predict_scaled(x_train, mc_samples=mc_samples)
            tr_loss = float(np.mean((train_pred - y_train) ** 2))
            val_pred, _ = self._predict_scaled(x_val, mc_samples=mc_samples)
            v_loss = float(np.mean((val_pred - y_val) ** 2))
            train_losses.append(tr_loss)
            val_losses.append(v_loss)

            if v_loss + 1e-8 < best_val:
                best_val = v_loss
                best_state = pyro.get_param_store().get_state()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stopping_rounds:
                    break

        if best_state is not None:
            pyro.get_param_store().set_state(best_state)

        return {
            "epochs_run": len(train_losses),
            "epoch_train_losses": train_losses,
            "epoch_val_losses": val_losses,
            "best_val_loss": float(best_val),
        }

    def _predict_scaled(
        self,
        x_scaled: np.ndarray,
        *,
        mc_samples: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        x_tensor = torch.from_numpy(np.asarray(x_scaled, dtype=np.float32))
        predictive = Predictive(
            self.model,
            guide=self.guide,
            num_samples=int(mc_samples),
            return_sites=["_RETURN"],
        )
        with torch.no_grad():
            samples = predictive(x_tensor)["_RETURN"]
        arr = samples.cpu().numpy()
        return arr.mean(axis=0), arr.std(axis=0)

    def predict(
        self,
        x: np.ndarray,
        *,
        mc_samples: int = 30,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.x_scaler is None or self.y_scaler is None:
            raise RuntimeError("ForwardBNNRegressor must be fitted before predict")
        x_scaled = self.x_scaler.transform(np.asarray(x, dtype=np.float32))
        mean_s, std_s = self._predict_scaled(x_scaled, mc_samples=mc_samples)
        mean = self.y_scaler.inverse_transform(mean_s)
        std = std_s * self.y_scaler.scale
        return mean, std

    def save(self, path: str | Path) -> None:
        if self.x_scaler is None or self.y_scaler is None:
            raise RuntimeError("Cannot save an unfitted ForwardBNNRegressor")
        torch.save(
            {
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "hidden_layers": self.hidden_layers,
                "prior_std": self.prior_std,
                "seed": self.seed,
                "x_scaler": self.x_scaler.to_dict(),
                "y_scaler": self.y_scaler.to_dict(),
                "param_store_state": pyro.get_param_store().get_state(),
            },
            Path(path),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ForwardBNNRegressor":
        ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
        regressor = cls(
            input_dim=int(ckpt["input_dim"]),
            output_dim=int(ckpt["output_dim"]),
            hidden_layers=list(ckpt["hidden_layers"]),
            prior_std=float(ckpt["prior_std"]),
            seed=int(ckpt.get("seed", 42)),
        )
        regressor.x_scaler = StandardScaler.from_dict(ckpt["x_scaler"])
        regressor.y_scaler = StandardScaler.from_dict(ckpt["y_scaler"])
        pyro.clear_param_store()
        pyro.get_param_store().set_state(ckpt["param_store_state"])
        return regressor


__all__ = [
    "ForwardBayesianMLP",
    "ForwardBNNRegressor",
]
