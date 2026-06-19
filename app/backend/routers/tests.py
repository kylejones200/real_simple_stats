import numpy as np
from fastapi import APIRouter, HTTPException
from backend.registry import TESTS
from backend.schemas import RunRequest
from backend.serializers import to_json_safe, explained_to_dict

router = APIRouter()

_STAT_FNS = {
    "mean": np.mean,
    "median": np.median,
    "std": np.std,
}


def _coerce_params(test_name: str, raw: dict) -> dict:
    """Convert JSON-decoded params to the shapes each function expects."""
    meta = TESTS[test_name]
    param_defs = {p["name"]: p for p in meta["params"]}
    coerced = {}

    for param in meta["params"]:
        name = param["name"]
        ptype = param["type"]
        val = raw.get(name, param.get("default"))

        if val is None and not param.get("required", True):
            continue  # omit optional None — let function use its own default

        if ptype == "array":
            coerced[name] = [float(x) for x in val]
        elif ptype == "array2d":
            coerced[name] = [[float(x) for x in row] for row in val]
        elif ptype == "multi_array":
            # groups is a list-of-lists; unpack as positional *args later
            coerced[name] = [[float(x) for x in g] for g in val]
        elif ptype == "float":
            coerced[name] = float(val) if val is not None else val
        elif ptype == "int":
            coerced[name] = int(val) if val is not None else val
        elif ptype == "bool":
            coerced[name] = bool(val)
        elif ptype == "str":
            if name == "statistic_fn" and val in _STAT_FNS:
                coerced[name] = _STAT_FNS[val]
            else:
                coerced[name] = str(val) if val is not None else val
        else:
            coerced[name] = val

    return coerced


def _call_fn(test_name: str, coerced: dict):
    meta = TESTS[test_name]
    fn = meta["fn"]

    # ANOVA-style: groups param is unpacked as positional *args
    if "groups" in coerced:
        groups = coerced.pop("groups")
        return fn(*groups, **coerced)

    return fn(**coerced)


@router.post("/explain/{test_name}")
def run_explained(test_name: str, body: RunRequest):
    if test_name not in TESTS:
        raise HTTPException(status_code=404, detail=f"Test '{test_name}' not found")
    if not TESTS[test_name]["explained"]:
        raise HTTPException(status_code=400, detail=f"'{test_name}' has no explained wrapper")

    try:
        coerced = _coerce_params(test_name, body.params)
        result = _call_fn(test_name, coerced)
        return {"result": explained_to_dict(result)}
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.post("/run/{test_name}")
def run_raw(test_name: str, body: RunRequest):
    if test_name not in TESTS:
        raise HTTPException(status_code=404, detail=f"Test '{test_name}' not found")

    try:
        coerced = _coerce_params(test_name, body.params)
        result = _call_fn(test_name, coerced)
        return {"result": to_json_safe(result)}
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
