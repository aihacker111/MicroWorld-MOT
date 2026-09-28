# ruff: noqa: E402 -- make direct `python tools/...` execution work without installation.

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from microworld_mot.cli import train_main

if __name__ == "__main__":
    train_main()
