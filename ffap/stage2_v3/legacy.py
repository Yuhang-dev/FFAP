from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import stage2_gate_causal_v2 as v2  # noqa: E402
import stage2_task_causal_gate as task_gate  # noqa: E402

__all__ = ["task_gate", "v2"]
