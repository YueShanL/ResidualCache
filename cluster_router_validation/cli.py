from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .factories import instantiate
from .metrics import MetricConfig, evaluate_validation_states
from .runner import ValidationRunConfig, collect_validation_states


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("configuration root must be an object")
    return payload


def _collect(config_path: Path) -> dict[str, Any]:
    payload = _load_json(config_path)
    expected = {"dataset", "model", "output_dir", "run"}
    unknown = set(payload).difference(expected)
    if unknown:
        raise ValueError(f"unknown collect configuration fields: {sorted(unknown)}")
    dataset = instantiate(payload["dataset"])
    model = instantiate(payload["model"])
    return collect_validation_states(
        dataset,
        model,
        payload["output_dir"],
        ValidationRunConfig(**dict(payload.get("run", {}))),
    )


def _metrics(state_dir: Path, output_dir: Path, config_path: Path | None) -> dict[str, Any]:
    payload = {} if config_path is None else _load_json(config_path)
    return evaluate_validation_states(
        state_dir,
        output_dir,
        MetricConfig(**payload),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect and score clustered-router end-to-end validation state"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("collect", help="run model comparisons and collect state")
    collect.add_argument("--config", type=Path, required=True)
    metrics = commands.add_parser("metrics", help="compute metrics from collected state")
    metrics.add_argument("--state-dir", type=Path, required=True)
    metrics.add_argument("--output-dir", type=Path, required=True)
    metrics.add_argument("--config", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "collect":
        result = _collect(arguments.config)
    else:
        result = _metrics(
            arguments.state_dir, arguments.output_dir, arguments.config
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
