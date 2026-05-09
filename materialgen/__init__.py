"""MaterialGen: concrete mix design with staged inverse models.

Each stage exposes a single ``run_*`` function; see ``materialgen/cli.py``
for the CLI wiring.

Public symbols load lazily so ``import materialgen`` does not import PyTorch/Pyro
until a symbol is actually accessed.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "run_evaluate_metrics",
    "run_make_neat_to_bnn",
    "run_train_forward",
    "run_train_gan",
    "run_train_neat",
    "run_validate_gost",
]

_LAZY = {
    "run_evaluate_metrics": ("evaluate_metrics", "run_evaluate_metrics"),
    "run_make_neat_to_bnn": ("make_neat_to_bnn", "run_make_neat_to_bnn"),
    "run_train_forward": ("train_forward", "run_train_forward"),
    "run_train_gan": ("train_gan", "run_train_gan"),
    "run_train_neat": ("train_neat", "run_train_neat"),
    "run_validate_gost": ("validate_gost", "run_validate_gost"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        module_name, attr = _LAZY[name]
        module = importlib.import_module(f".{module_name}", __package__)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
