from __future__ import annotations

import argparse
import json
from pathlib import Path

from .make_neat_to_bnn import run_make_neat_to_bnn
from .train_neat import run_train_neat
from .train_gan import run_train_gan


def _write_payload(payload: str, output_path: str | None) -> None:
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
    print(payload)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="materialgen",
        description="Concrete mix design: inverse NEAT training and BNN fine-tuning.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    neat_parser = subparsers.add_parser(
        "train_neat",
        help="Train the inverse NEAT network and save it under artifacts/train_neat",
    )
    neat_parser.add_argument("--config", required=True, help="Path to backward.json")
    neat_parser.add_argument("--artifacts-dir", default="artifacts", help="Root directory for all stage artifacts")
    neat_parser.add_argument("--inverse-dir", default=None, help="Optional override for the train_neat artifacts folder")
    neat_parser.add_argument("--output", default=None, help="Optional path for JSON summary")

    bnn_parser = subparsers.add_parser(
        "make_neat_to_bnn",
        help="Convert trained NEAT network into a Bayesian NN and fine-tune on known data",
    )
    bnn_parser.add_argument("--config", required=True, help="Path to make_neat_to_bnn.json")
    bnn_parser.add_argument("--artifacts-dir", default="artifacts", help="Root directory for all stage artifacts")
    bnn_parser.add_argument("--inverse-dir", default=None, help="Optional override for the train_neat artifacts folder")
    bnn_parser.add_argument("--bnn-dir", default=None, help="Optional override for the make_neat_to_bnn artifacts folder")
    bnn_parser.add_argument("--output", default=None, help="Optional path for JSON summary")

    gan_parser = subparsers.add_parser(
        "train_gan",
        help="Train GAN discriminator for validation of realism of predictions",
    )
    gan_parser.add_argument("--config", required=True, help="Path to train_gan.json")
    gan_parser.add_argument("--artifacts-dir", default="artifacts", help="Root directory for all stage artifacts")
    gan_parser.add_argument("--bnn-dir", default=None, help="Optional override for the BNN artifacts folder")
    gan_parser.add_argument("--gan-dir", default=None, help="Optional override for the train_gan artifacts folder")
    gan_parser.add_argument("--output", default=None, help="Optional path for JSON summary")

    return parser


def _handle_train_neat(args) -> int:
    summary = run_train_neat(
        config_path=args.config,
        artifacts_dir=args.artifacts_dir,
        inverse_dir=args.inverse_dir,
    )
    _write_payload(json.dumps(summary, ensure_ascii=False, indent=2), args.output)
    return 0


def _handle_make_neat_to_bnn(args) -> int:
    summary = run_make_neat_to_bnn(
        config_path=args.config,
        artifacts_dir=args.artifacts_dir,
        inverse_dir=args.inverse_dir,
        bnn_dir=args.bnn_dir,
    )
    _write_payload(json.dumps(summary, ensure_ascii=False, indent=2), args.output)
    return 0


def _handle_train_gan(args) -> int:
    summary = run_train_gan(
        config_path=args.config,
        artifacts_dir=args.artifacts_dir,
        bnn_dir=args.bnn_dir,
        gan_dir=args.gan_dir,
    )
    _write_payload(json.dumps(summary, ensure_ascii=False, indent=2), args.output)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    handlers = {
        "train_neat": _handle_train_neat,
        "make_neat_to_bnn": _handle_make_neat_to_bnn,
        "train_gan": _handle_train_gan,
    }
    handler = handlers.get(args.command)
    if handler is None:
        parser.error(f"Unknown command: {args.command}")
        return 2
    return handler(args)
