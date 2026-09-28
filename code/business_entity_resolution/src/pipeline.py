"""Backwards-compatibility proxy for pipeline.py."""
import sys
from pathlib import Path

ROOT_SRC = Path(__file__).resolve().parents[3] / "src"
if str(ROOT_SRC) not in sys.path:
    sys.path.insert(0, str(ROOT_SRC))

from pipeline import main, log, run_block, apply_topk, labels_for, get_tf_index

if __name__ == "__main__":
    main()
