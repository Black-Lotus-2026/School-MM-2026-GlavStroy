"""Локальный помощник: последовательный запуск всех стадий из корня репозитория.

Запускать из каталога проекта:
  python main_work.py

Нужны установленные зависимости и заранее подготовленный CSV (см. examples/*.json).
"""

from __future__ import annotations

import materialgen.cli as cl

# Аргументы для argparse; порядок = порядок пайплайна.
_PIPELINE_ARGV = (
    (
        "train_neat",
        "--config",
        "examples/backward.json",
        "--artifacts-dir",
        "artifacts",
    ),
    (
        "make_neat_to_bnn",
        "--config",
        "examples/make_neat_to_bnn.json",
        "--artifacts-dir",
        "artifacts",
    ),
    (
        "train_gan",
        "--config",
        "examples/train_gan.json",
        "--artifacts-dir",
        "artifacts",
    ),
    (
        "evaluate_metrics",
        "--config",
        "examples/evaluate_metrics.json",
    ),
)


def main() -> int:
    parser = cl._build_parser()
    handlers = {
        "train_neat": cl._handle_train_neat,
        "make_neat_to_bnn": cl._handle_make_neat_to_bnn,
        "train_gan": cl._handle_train_gan,
        "evaluate_metrics": cl._handle_evaluate_metrics,
    }
    exit_code = 0
    for argv in _PIPELINE_ARGV:
        args = parser.parse_args(list(argv))
        code = handlers[args.command](args)
        if code:
            exit_code = code
            break
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
