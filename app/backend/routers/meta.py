from fastapi import APIRouter, HTTPException
from backend.registry import TESTS, MODULE_GROUPS

router = APIRouter()


@router.get("/tests")
def list_tests():
    result = []
    for name, meta in TESTS.items():
        result.append({
            "name": name,
            "label": meta["label"],
            "module": meta["module"],
            "explained": meta["explained"],
            "params": meta["params"],
            "has_sample_data": meta.get("sample_data") is not None,
        })
    return result


@router.get("/tests/{test_name}/schema")
def get_test_schema(test_name: str):
    if test_name not in TESTS:
        raise HTTPException(status_code=404, detail=f"Test '{test_name}' not found")
    meta = TESTS[test_name]
    return {
        "name": test_name,
        "label": meta["label"],
        "module": meta["module"],
        "explained": meta["explained"],
        "params": meta["params"],
        "sample_data": meta.get("sample_data"),
    }


@router.get("/modules")
def list_modules():
    return MODULE_GROUPS
