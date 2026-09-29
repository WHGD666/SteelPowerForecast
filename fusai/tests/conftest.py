"""Make the isolated fusai package importable during local tests."""

from __future__ import annotations

import sys
from pathlib import Path


FUSAI_ROOT = Path(__file__).resolve().parents[1]
if str(FUSAI_ROOT) not in sys.path:
    sys.path.insert(0, str(FUSAI_ROOT))
