"""Backward-compatible alias for the countdown eval.

The canonical implementation now lives in ``countdown_eval.py`` (the module name used in
the run commands). This module re-exports it so existing
``python -m DiffGemma.diffgemma_trl.sft_eval_countdown`` invocations keep working.
"""

from __future__ import annotations

from .countdown_eval import _score_countdown, main

__all__ = ["_score_countdown", "main"]


if __name__ == "__main__":
    main()
