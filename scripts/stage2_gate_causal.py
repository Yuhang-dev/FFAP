"""Compatibility entry point for the frozen Stage 2 v1 gate."""

from stage2_gate_causal_v1 import *  # noqa: F401,F403
from stage2_gate_causal_v1 import main


if __name__ == "__main__":
    raise SystemExit(main())
