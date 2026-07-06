"""Display helpers (pillar: base).

Two display modes share one formatting vocabulary:

  * COMPACT (default) — the one/two-line summaries the callbacks log per rollout
    iteration / eval: ``fmt`` / ``lead`` / ``count`` scalar formatting and the
    ``proven_by_depth`` line builder.
  * DIAGNOSIS (opt-in, ``log_diagnostics``) — ``print_formatted_metrics``, the full
    sorted key/value table over every collected metric (per-depth, per-predicate,
    per-length, terminal categories).

The metric VALUES themselves are produced by the collectors (``_metrics.py`` /
``kge/callbacks/metrics.py``) — breakdown entries are pre-formatted
``'<mean> +/- <std> (<count>)'`` strings (``_format_stat_string``); the display
layer only renders them.
"""

import logging
import re
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class Display:
    """Formatting helpers for the compact summaries + the diagnosis table."""

    # ── scalar / stat-string formatting ───────────────────────────────

    @staticmethod
    def _format_stat_string(mean: Optional[float], std: Optional[float], count: int) -> str:
        """Format statistics as a display string."""
        if mean is None or std is None or count == 0:
            return "N/A"
        return f"{mean:.3f} +/- {std:.2f} ({count})"

    @staticmethod
    def fmt(v: Any, nd: int = 3) -> str:
        """A scalar for a compact line; ``—`` when absent."""
        if v is None:
            return "—"
        if isinstance(v, (float, np.floating)):
            return f"{v:.{nd}f}"
        return str(v)

    @staticmethod
    def lead(v: Any) -> Optional[float]:
        """The leading float of a stat string like ``'0.440 +/- 0.49 (500)'`` (or a number)."""
        if isinstance(v, (int, float, np.number)):
            return float(v)
        if not isinstance(v, str):
            return None
        m = re.match(r"\s*([-0-9.]+)", v)
        return float(m.group(1)) if m else None

    @staticmethod
    def count(v: Any) -> Optional[int]:
        """The trailing episode count of a stat string like ``'0.440 +/- 0.49 (500)'``."""
        if not isinstance(v, str):
            return None
        m = re.search(r"\((\d+)\)\s*$", v)
        return int(m.group(1)) if m else None

    @staticmethod
    def _format_depth_key(depth_value: Any) -> str:
        """Normalize depth IDs."""
        if depth_value == -1 or depth_value is None:
            return "unknown"
        return str(depth_value)

    # ── compact line builders ─────────────────────────────────────────

    @staticmethod
    def proven_by_depth(metrics: Dict[str, Any], label: str = "pos") -> str:
        """``proven_d_<d>_<label>`` entries as one line: ``d1 1.000/200 d2 0.895/19 d? 0.004/256``
        (value/episode-count; ``d?`` = no annotated gold depth). Empty string when absent."""
        items = []
        for k, v in metrics.items():
            m = re.match(rf"proven_d_(\w+)_{label}$", k)
            if m:
                d = m.group(1)
                mean, n = Display.lead(v), Display.count(v)
                items.append((d, mean, n))
        items.sort(key=lambda t: float("inf") if t[0] == "unknown" else int(t[0]))
        return " ".join(
            f"d{'?' if d == 'unknown' else d} {Display.fmt(mean)}/{n if n is not None else '?'}"
            for d, mean, n in items)

    # ── the diagnosis table (opt-in) ──────────────────────────────────

    @staticmethod
    def print_formatted_metrics(
        metrics: Dict[str, Any],
        prefix: str = "rollout",
        extra_metrics: Optional[Dict[str, Any]] = None,
        global_step: Optional[int] = None,
    ) -> None:
        """The full sorted key/value table over every collected metric (diagnosis mode)."""
        final_output = {}
        for source in (metrics, extra_metrics):
            if source:
                for k, v in source.items():
                    if isinstance(v, (float, np.floating)):
                        final_output[k] = f"{v:.3f}"
                    else:
                        final_output[k] = str(v)
        if global_step is not None and "total_timesteps" not in final_output:
            final_output["total_timesteps"] = str(global_step)

        # Fixed widths to accommodate longest metric names (e.g., proven_d_unknown_neg_brother)
        key_width = 35
        val_width = 26
        line_width = key_width + val_width + 7  # 7 for "| " + " | " + " |"

        print("-" * line_width)
        if final_output:
            print(f"| {prefix + '/':<{key_width-1}} | {'':<{val_width}} |")
            for key in sorted(final_output.keys()):
                print(f"|    {key:<{key_width-5}} | {final_output[key]:<{val_width}} |")
        print("-" * line_width)
        print()
