from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routers import meta, tests, plots

app = FastAPI(title="real_simple_stats API", version="0.4.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

app.include_router(meta.router,   prefix="/api")
app.include_router(tests.router,  prefix="/api")
app.include_router(plots.router,  prefix="/api")


@app.get("/api/health")
def health():
    return {"status": "ok", "library": "real_simple_stats"}
