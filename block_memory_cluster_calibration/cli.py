from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runner import run_calibration, validate_config


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate independent router-key block memory over one fixed GPU "
            "block cache per sample."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--validate-only", action="store_true")
    arguments = parser.parse_args(argv)
    path = Path(arguments.config).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("calibration config root must be an object")
    if arguments.validate_only:
        result = validate_config(config)
        print(
            json.dumps(
                {
                    "valid": True,
                    "config": str(path),
                    **{
                        key: str(value) if isinstance(value, Path) else value
                        for key, value in result.items()
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    run_calibration(config)
    return 0


__all__ = ["main"]
