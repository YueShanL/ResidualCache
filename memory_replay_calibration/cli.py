from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runner import run_calibration


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate cluster-local usage eviction by full memory replay."
    )
    parser.add_argument("--config", required=True)
    arguments = parser.parse_args(argv)
    path = Path(arguments.config).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("calibration config root must be an object")
    run_calibration(config)
    return 0
