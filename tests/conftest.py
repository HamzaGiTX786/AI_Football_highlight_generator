"""Shared pytest fixtures and path setup."""

import sys
from pathlib import Path

# Make the package importable when running `pytest` from the project root.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
