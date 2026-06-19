from pydantic import BaseModel
from typing import Any


class RunRequest(BaseModel):
    params: dict[str, Any]


class RunResponse(BaseModel):
    result: dict[str, Any]


class PlotResponse(BaseModel):
    png: str


class ParamSchema(BaseModel):
    name: str
    type: str
    label: str
    default: Any = None
    required: bool = True


class TestMeta(BaseModel):
    name: str
    label: str
    module: str
    explained: bool
    params: list[ParamSchema]
    sample_data: dict[str, Any] | None = None
