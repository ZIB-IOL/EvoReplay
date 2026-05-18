"""Vendored from skydiscover.utils.metrics so the static-analysis path has no
hard runtime dependency on skydiscover. Kept verbatim except for the imports
trimmed down to what get_score actually uses.
"""

from typing import Any, Dict


def is_numeric_metric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def get_score(metrics: Dict[str, Any]) -> float:
    """Return combined_score if available, otherwise average of all numeric metric values."""
    if not metrics:
        return 0.0
    if "combined_score" in metrics:
        try:
            return float(metrics["combined_score"])
        except (ValueError, TypeError):
            pass
    numeric_values = [v for v in metrics.values() if is_numeric_metric(v)]
    return sum(numeric_values) / len(numeric_values) if numeric_values else 0.0
