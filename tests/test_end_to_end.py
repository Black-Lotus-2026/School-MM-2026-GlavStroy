"""Smoke tests: package imports cleanly and CLI parser builds."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from materialgen.cli import _build_parser, main as cli_main


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = REPO_ROOT / "examples"


class CliSmokeTest(unittest.TestCase):
    def test_all_subcommands_register(self) -> None:
        """The CLI exposes all pipeline commands."""

        parser = _build_parser()
        # argparse stores subcommands in the choices of the subparsers action.
        sub_action = next(a for a in parser._subparsers._actions if a.dest == "command")
        self.assertEqual(
            set(sub_action.choices),
            {
                "train_neat",
                "make_neat_to_bnn",
                "train_gan",
                "evaluate_metrics",
                "train_forward",
                "validate_gost",
            },
        )

    def test_each_stage_has_example_config(self) -> None:
        """Checked-in example JSON exists for each CLI command."""

        for name in (
            "backward.json",
            "make_neat_to_bnn.json",
            "train_gan.json",
            "train_gan_forward.json",
            "evaluate_metrics.json",
            "forward_normal.json",
            "forward_synthetic.json",
            "validate_gost.json",
        ):
            path = EXAMPLES / name
            self.assertTrue(path.exists(), f"missing example: {path}")
            json.loads(path.read_text(encoding="utf-8"))  # parses as JSON

    def test_help_runs(self) -> None:
        """`main.py --help` exits cleanly without traceback."""

        with self.assertRaises(SystemExit) as exc:
            cli_main(["--help"])
        self.assertEqual(exc.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
