import sys
from pathlib import Path


# Ensure the package in src/ is importable without installation
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SRC_STR = str(SRC)
if SRC_STR not in sys.path:
    sys.path.insert(0, SRC_STR)
