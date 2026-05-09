"""Стадия 4 — дискриминатор «реальные пары vs NEAT-BNN».

Обученная NEAT→BNN модель задаёт **генератор** в смысле GAN: по целевым
свойствам (прочность и т.д.) предсказывает состав смеси. Замороженная BNN
сравнивается с реальными парами (состав, свойства); дискриминатор учится
отличать реальные записи от предсказаний BNN.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from pyro.nn import PyroModule, PyroSample
import pyro
import pyro.distributions as dist

from .config import DatasetInputConfig, OptimizerConfig, _resolve_config_path
from .data import prepare_dataset
from .neat_bnn import NeatBNNRegressor
from .scaler import StandardScaler
from .stage_common import resolve_artifacts_layout, write_json


# =============================================================================
# Конфигурация стадии GAN
# =============================================================================

@dataclass
class GANStageConfig:
    """JSON configuration used by the GAN training stage."""

    dataset: DatasetInputConfig
    discriminator_config: OptimizerConfig = field(default_factory=OptimizerConfig)
    
    # GAN-specific hyperparameters
    num_epochs: int = 100
    batch_size: int = 32
    discriminator_learning_rate: float = 0.001
    generator_learning_rate: float = 0.0001
    adversarial_weight: float = 1.0
    reconstruction_weight: float = 0.5
    constraint_weight: float = 0.1
    warmup_epochs: int = 5
    
    # Discriminator architecture
    discriminator_hidden_layers: list[int] = field(default_factory=lambda: [64, 32])
    use_spectral_norm: bool = False
    
    # GAN training strategy
    discriminator_steps_per_epoch: int = 1
    generator_steps_per_epoch: int = 1
    gradient_penalty_weight: float = 10.0  # For WGAN-GP stability
    
    # Regularization
    use_focal_loss: bool = False
    focal_loss_gamma: float = 2.0
    label_smoothing: float = 0.0
    
    neat_config_path: str | None = None
    # Loaded from ``artifacts/make_neat_to_bnn`` by default
    bnn_model_filename: str = "bnn_model.pt"
    generator_mc_samples: int = 30
    random_seed: int | None = None

    @property
    def component_columns(self) -> list[str]:
        return self.dataset.components

    @property
    def property_columns(self) -> list[str]:
        return self.dataset.properties

    @classmethod
    def from_dict(cls, payload: dict) -> "GANStageConfig":
        """Load GAN config from JSON."""
        dataset_payload = payload.get("dataset", payload.get("gan_input", {}))
        
        dataset = DatasetInputConfig.from_dict(
            dataset_payload,
            legacy_data_path=payload.get("data_path"),
            legacy_components=payload.get("components"),
            legacy_properties=payload.get("properties"),
        )
        
        discriminator_config = OptimizerConfig.from_dict(
            payload.get("discriminator_config", {})
        )
        
        gan_params = {
            "dataset": dataset,
            "discriminator_config": discriminator_config,
        }
        
        for key in [
            "num_epochs", "batch_size", "discriminator_learning_rate",
            "generator_learning_rate", "adversarial_weight", "reconstruction_weight",
            "constraint_weight", "warmup_epochs", "discriminator_hidden_layers",
            "use_spectral_norm", "discriminator_steps_per_epoch",
            "generator_steps_per_epoch", "gradient_penalty_weight",
            "use_focal_loss", "focal_loss_gamma", "label_smoothing",
            "neat_config_path",
            "bnn_model_filename",
            "generator_mc_samples",
            "random_seed",
        ]:
            if key in payload:
                gan_params[key] = payload[key]
        
        return cls(**gan_params)

    def resolve_paths(self, base_dir: Path) -> None:
        """Resolve relative paths."""
        self.dataset.resolve_paths(base_dir)
        resolved = _resolve_config_path(base_dir, self.neat_config_path)
        self.neat_config_path = resolved

    def validate(self) -> None:
        """Validate configuration."""
        self.dataset.validate("dataset")
        if self.num_epochs < 1:
            raise ValueError("num_epochs must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.generator_mc_samples < 1:
            raise ValueError("generator_mc_samples must be >= 1")
        if not str(self.bnn_model_filename).strip():
            raise ValueError("bnn_model_filename must not be empty")

    def to_dict(self) -> dict:
        """Serialize to JSON."""
        return {
            "dataset": self.dataset.to_dict(),
            "discriminator_config": self.discriminator_config.__dict__,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "discriminator_learning_rate": self.discriminator_learning_rate,
            "generator_learning_rate": self.generator_learning_rate,
            "adversarial_weight": self.adversarial_weight,
            "reconstruction_weight": self.reconstruction_weight,
            "constraint_weight": self.constraint_weight,
            "warmup_epochs": self.warmup_epochs,
            "discriminator_hidden_layers": self.discriminator_hidden_layers,
            "use_spectral_norm": self.use_spectral_norm,
            "discriminator_steps_per_epoch": self.discriminator_steps_per_epoch,
            "generator_steps_per_epoch": self.generator_steps_per_epoch,
            "gradient_penalty_weight": self.gradient_penalty_weight,
            "use_focal_loss": self.use_focal_loss,
            "focal_loss_gamma": self.focal_loss_gamma,
            "label_smoothing": self.label_smoothing,
            "bnn_model_filename": self.bnn_model_filename,
            "generator_mc_samples": self.generator_mc_samples,
            "random_seed": self.random_seed,
        }


def _alignment_check(manifest: dict, dataset: DatasetInputConfig) -> None:
    """BNN manifest column order must match the GAN dataset block exactly."""

    mi = list(manifest.get("input_names", []))
    mo = list(manifest.get("output_names", []))
    if mi != list(dataset.properties):
        raise ValueError(
            "GAN dataset.properties must match make_neat_to_bnn manifest input_names "
            f"(same order): manifest={mi}, config={list(dataset.properties)}"
        )
    if mo != list(dataset.components):
        raise ValueError(
            "GAN dataset.components must match make_neat_to_bnn manifest output_names "
            f"(same order): manifest={mo}, config={list(dataset.components)}"
        )


# =============================================================================
# Discriminator Architecture
# =============================================================================

class BayesianDiscriminator(PyroModule):
    """Bayesian Neural Network Discriminator.
    
    Использует Pyro для моделирования неопределённости весов.
    Выходной слой даёт вероятность того, что примеры реальные.
    """

    def __init__(
        self,
        input_size: int,
        hidden_layers: list[int],
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.use_spectral_norm = use_spectral_norm

        layers = []
        prev_size = input_size

        # Hidden layers with Bayesian weights
        for hidden_size in hidden_layers:
            layers.append(
                PyroModule[nn.Linear](prev_size, hidden_size)
            )
            layers.append(nn.ReLU())
            prev_size = hidden_size

        # Output layer: single output for binary classification
        layers.append(PyroModule[nn.Linear](prev_size, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

        # Initialize Bayesian priors
        self._init_priors()

    def _init_priors(self):
        """Initialize Pyro priors for all Linear layers."""
        for layer in self.network:
            if isinstance(layer, PyroModule[nn.Linear]):
                layer.weight = PyroSample(
                    dist.Normal(0., 1.).expand([layer.out_features, layer.in_features]).to_event(2)
                )
                layer.bias = PyroSample(
                    dist.Normal(0., 1.).expand([layer.out_features]).to_event(1)
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through discriminator.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Probability tensor of shape (batch_size, 1)
        """
        return self.network(x)


class DeterministicDiscriminator(nn.Module):
    """Детерминированный дискриминатор для быстрого обучения."""

    def __init__(
        self,
        input_size: int,
        hidden_layers: list[int],
        use_spectral_norm: bool = False,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        layers = []
        prev_size = input_size

        for hidden_size in hidden_layers:
            linear = nn.Linear(prev_size, hidden_size)
            if use_spectral_norm:
                linear = nn.utils.spectral_norm(linear)
            layers.append(linear)
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        # Output layer
        output_layer = nn.Linear(prev_size, 1)
        if use_spectral_norm:
            output_layer = nn.utils.spectral_norm(output_layer)
        layers.append(output_layer)
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)


# =============================================================================
# GAN Trainer
# =============================================================================

class GANTrainer:
    """Trainer for GAN with Discriminator validation."""

    def __init__(
        self,
        generator_fn,  # функция для генерации предсказаний
        component_bounds_lower: np.ndarray,
        component_bounds_upper: np.ndarray,
        config: GANStageConfig,
        input_scaler: StandardScaler | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.generator_fn = generator_fn
        self.component_bounds_lower = torch.from_numpy(component_bounds_lower).float().to(device)
        self.component_bounds_upper = torch.from_numpy(component_bounds_upper).float().to(device)
        self.input_scaler = input_scaler
        if input_scaler is not None:
            self._scaler_mean = torch.from_numpy(np.asarray(input_scaler.mean, dtype=np.float32)).to(device)
            self._scaler_scale = torch.from_numpy(np.asarray(input_scaler.scale, dtype=np.float32)).to(device)
        else:
            self._scaler_mean = None
            self._scaler_scale = None
        self.config = config
        self.device = device

        # Create discriminator
        input_size = len(config.component_columns) + len(config.property_columns)
        self.discriminator = DeterministicDiscriminator(
            input_size=input_size,
            hidden_layers=config.discriminator_hidden_layers,
            use_spectral_norm=config.use_spectral_norm,
        ).to(device)

        # Optimizers
        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=config.discriminator_learning_rate,
            betas=(0.5, 0.999),
        )

        self.history: dict[str, list[float]] = {}

    def _apply_constraints(self, components: torch.Tensor) -> torch.Tensor:
        """Clip components to valid bounds."""
        return torch.clamp(components, self.component_bounds_lower, self.component_bounds_upper)

    def _scale_input(self, combined: torch.Tensor) -> torch.Tensor:
        """Standardize discriminator input (components, properties) so BCELoss does not saturate."""
        if self._scaler_mean is None:
            return combined
        return (combined - self._scaler_mean) / self._scaler_scale

    def _compute_focal_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        gamma: float = 2.0,
    ) -> torch.Tensor:
        """Focal loss for addressing class imbalance."""
        bce = nn.BCELoss(reduction="none")
        loss = bce(logits, targets)
        p_t = torch.where(targets == 1, logits, 1 - logits)
        focal_weight = (1 - p_t) ** gamma
        return (focal_weight * loss).mean()

    def _compute_discriminator_loss(
        self,
        real_data: torch.Tensor,
        fake_data: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Compute discriminator loss."""
        
        # Forward pass
        real_pred = self.discriminator(real_data)
        fake_pred = self.discriminator(fake_data)

        # Labels
        real_labels = torch.ones_like(real_pred) - self.config.label_smoothing
        fake_labels = torch.zeros_like(fake_pred) + self.config.label_smoothing

        # Loss
        if self.config.use_focal_loss:
            loss_real = self._compute_focal_loss(
                real_pred, real_labels, self.config.focal_loss_gamma
            )
            loss_fake = self._compute_focal_loss(
                fake_pred, fake_labels, self.config.focal_loss_gamma
            )
        else:
            loss_real = nn.BCELoss()(real_pred, real_labels)
            loss_fake = nn.BCELoss()(fake_pred, fake_labels)

        d_loss = loss_real + loss_fake

        # Metrics
        acc_real = (real_pred > 0.5).float().mean()
        acc_fake = (fake_pred < 0.5).float().mean()

        return d_loss, {
            "d_loss": d_loss.item(),
            "acc_real": acc_real.item(),
            "acc_fake": acc_fake.item(),
        }

    def train_epoch(
        self,
        real_data: np.ndarray,
        properties: np.ndarray,
    ) -> dict:
        """Train one epoch of GAN."""
        
        n_samples = len(real_data)
        batch_size = self.config.batch_size
        
        # Create batches
        indices = np.random.permutation(n_samples)
        batches = [
            indices[i : i + batch_size]
            for i in range(0, n_samples, batch_size)
        ]

        epoch_metrics = {
            "d_loss": [],
            "acc_real": [],
            "acc_fake": [],
        }

        for batch_idx in batches:
            # Get batch
            batch_real = torch.from_numpy(real_data[batch_idx]).float().to(self.device)
            batch_props = torch.from_numpy(properties[batch_idx]).float().to(self.device)

            # Generate fake data using generator
            with torch.no_grad():
                batch_fake = self.generator_fn(batch_props)
                batch_fake = self._apply_constraints(batch_fake)

            # Combine and standardize: [components, properties]
            real_combined = self._scale_input(torch.cat([batch_real, batch_props], dim=1))
            fake_combined = self._scale_input(torch.cat([batch_fake, batch_props], dim=1))

            # Train discriminator
            self.d_optimizer.zero_grad()
            d_loss, metrics = self._compute_discriminator_loss(real_combined, fake_combined)
            d_loss.backward()
            self.d_optimizer.step()

            for key, value in metrics.items():
                if key in epoch_metrics:
                    epoch_metrics[key].append(value)

        # Average metrics
        return {key: np.mean(values) for key, values in epoch_metrics.items()}

    def evaluate(
        self,
        real_data: np.ndarray,
        properties: np.ndarray,
    ) -> dict:
        """Evaluate discriminator on validation set."""
        
        real_data = torch.from_numpy(real_data).float().to(self.device)
        properties = torch.from_numpy(properties).float().to(self.device)

        with torch.no_grad():
            fake_data = self.generator_fn(properties)
            fake_data = self._apply_constraints(fake_data)

            real_combined = self._scale_input(torch.cat([real_data, properties], dim=1))
            fake_combined = self._scale_input(torch.cat([fake_data, properties], dim=1))

            _, metrics = self._compute_discriminator_loss(real_combined, fake_combined)

        return metrics


# =============================================================================
# Main entry point
# =============================================================================

def load_gan_config(path: str | Path) -> GANStageConfig:
    """Load GAN configuration from JSON file."""
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = GANStageConfig.from_dict(payload)
    config.resolve_paths(path.parent)
    config.validate()
    return config


def run_train_gan(
    config_path: str | Path,
    artifacts_dir: str | Path = "artifacts",
    bnn_dir: str | Path | None = None,
    gan_dir: str | Path | None = None,
) -> dict:
    """Train a discriminator that separates real (composition, property) pairs from BNN predictions."""

    config = load_gan_config(config_path)

    layout = resolve_artifacts_layout(
        artifacts_dir=artifacts_dir,
        inverse_dir=None,
        bnn_dir=bnn_dir,
        gan_dir=gan_dir,
    )

    gan_artifacts = layout.gan_artifacts
    gan_artifacts.mkdir(parents=True, exist_ok=True)

    manifest_path = layout.bnn_dir / "bnn_model_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing BNN manifest {manifest_path}. Run make_neat_to_bnn before train_gan."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _alignment_check(manifest, config.dataset)

    model_path = layout.bnn_dir / config.bnn_model_filename
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing trained BNN checkpoint {model_path}. Run make_neat_to_bnn first."
        )

    regressor = NeatBNNRegressor.load(model_path)

    dataset = prepare_dataset(
        csv_path=config.dataset.data_path,
        component_columns=config.dataset.components,
        property_columns=config.dataset.properties,
    )

    seed = config.random_seed
    if seed is None:
        seed = config.discriminator_config.seed
    if seed is None:
        seed = 42
    rng = np.random.default_rng(int(seed))

    n_samples = len(dataset.components)
    n_train = int(0.8 * n_samples)
    if n_train < 1 or n_train >= n_samples:
        raise ValueError(
            f"Need at least 2 rows for train/val split; got n_samples={n_samples}"
        )

    indices = rng.permutation(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    train_components = dataset.components[train_idx]
    train_properties = dataset.properties[train_idx]
    val_components = dataset.components[val_idx]
    val_properties = dataset.properties[val_idx]

    bounds_lower = np.array(
        [dataset.component_bounds[name][0] for name in config.dataset.components],
        dtype=float,
    )
    bounds_upper = np.array(
        [dataset.component_bounds[name][1] for name in config.dataset.components],
        dtype=float,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mc_samples = max(1, int(config.generator_mc_samples))

    # Standardize discriminator input on the train fold so BCELoss does not
    # saturate (compositions ~1e2-1e3 kg/m^3, strength ~1e1 MPa).
    train_real_combined = np.concatenate([train_components, train_properties], axis=1)
    input_scaler = StandardScaler.fit(train_real_combined)

    def generator_fn(properties_tensor: torch.Tensor) -> torch.Tensor:
        props = properties_tensor.detach().cpu().numpy()
        mean, _ = regressor.predict_components(props, mc_samples=mc_samples)
        return torch.from_numpy(np.asarray(mean, dtype=np.float32)).to(device)

    trainer = GANTrainer(
        generator_fn=generator_fn,
        component_bounds_lower=bounds_lower,
        component_bounds_upper=bounds_upper,
        config=config,
        input_scaler=input_scaler,
        device=device,
    )

    log_every = max(1, config.num_epochs // 10)

    for epoch in range(config.num_epochs):
        metrics = trainer.train_epoch(train_components, train_properties)

        if (epoch + 1) % log_every == 0 or epoch == config.num_epochs - 1:
            val_metrics = trainer.evaluate(val_components, val_properties)
            print(
                f"Epoch {epoch + 1}/{config.num_epochs}: D_loss={metrics['d_loss']:.4f}, "
                f"Acc_real={metrics['acc_real']:.4f}, Acc_fake={metrics['acc_fake']:.4f} | "
                f"val D_loss={val_metrics['d_loss']:.4f}"
            )

        for key, value in metrics.items():
            trainer.history.setdefault(key, []).append(float(value))

    torch.save(trainer.discriminator.state_dict(), gan_artifacts / "discriminator.pt")

    write_json(gan_artifacts / "input_scaler.json", input_scaler.to_dict())
    write_json(
        gan_artifacts / "gan_generator_meta.json",
        {
            "bnn_checkpoint": str(model_path.resolve()),
            "generator_mc_samples": mc_samples,
            "manifest": str(manifest_path.resolve()),
            "input_scaler": "input_scaler.json",
        },
    )
    write_json(gan_artifacts / "history.json", trainer.history)

    summary = {
        "status": "success",
        "num_epochs": config.num_epochs,
        "bnn_model": str(model_path),
        "random_seed": int(seed),
        "final_d_loss": float(trainer.history["d_loss"][-1]),
        "final_acc_real": float(trainer.history["acc_real"][-1]),
        "final_acc_fake": float(trainer.history["acc_fake"][-1]),
        "artifacts_dir": str(gan_artifacts),
    }

    write_json(gan_artifacts / "train_gan.json", summary)

    return summary

