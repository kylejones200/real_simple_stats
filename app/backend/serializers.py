import numpy as np
from real_simple_stats.explain import ExplainedResult


def to_json_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, frozenset):
        return sorted(str(x) for x in obj)
    if callable(obj):
        return None
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]
    return obj


def explained_to_dict(r: ExplainedResult) -> dict:
    return {
        "title": r.title,
        "question": r.question,
        "values": to_json_safe(r.values),
        "intuition": r.intuition,
        "interpretation": r.interpretation,
        "assumptions": to_json_safe(r.assumptions),
        "caveats": list(r.caveats),
        "next_steps": list(r.next_steps),
        "decision": r.decision,
        "has_plot": r._plot_fn is not None,
    }
