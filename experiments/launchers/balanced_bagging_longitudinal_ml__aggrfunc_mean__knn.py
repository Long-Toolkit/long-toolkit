from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.engine import run_launcher_cli


if __name__ == "__main__":
    run_launcher_cli('balanced_bagging_longitudinal_ml__aggrfunc_mean__knn')
