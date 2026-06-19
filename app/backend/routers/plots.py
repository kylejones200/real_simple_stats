import io
import base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fastapi import APIRouter, HTTPException
from backend.registry import TESTS
from backend.schemas import RunRequest
from backend.routers.tests import _coerce_params, _call_fn

router = APIRouter()


@router.post("/plot/{test_name}")
def get_plot(test_name: str, body: RunRequest):
    if test_name not in TESTS:
        raise HTTPException(status_code=404, detail=f"Test '{test_name}' not found")
    if not TESTS[test_name]["explained"]:
        raise HTTPException(status_code=400, detail=f"'{test_name}' has no plot")

    try:
        coerced = _coerce_params(test_name, body.params)
        result = _call_fn(test_name, coerced)
        if result._plot_fn is None:
            raise HTTPException(status_code=404, detail="No plot for this result")

        plot_output = result._plot_fn()
        # plot functions return either fig or (fig, ax) or (fig, axes_tuple)
        fig = plot_output[0] if isinstance(plot_output, tuple) else plot_output
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        png_b64 = base64.b64encode(buf.read()).decode()
        plt.close(fig)
        return {"png": png_b64}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
