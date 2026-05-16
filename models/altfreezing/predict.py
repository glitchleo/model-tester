from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location("root_predict", ROOT / "predict.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Could not load root predict.py")
root_predict = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(root_predict)

if __name__ == "__main__":
    raise SystemExit(root_predict.main(default_model="altfreezing"))
