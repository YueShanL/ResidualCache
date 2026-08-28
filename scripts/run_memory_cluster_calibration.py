from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memory_cluster_calibration.cli import main

raise SystemExit(main())
