from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    sample_rate: int


class ErrorResponse(BaseModel):
    detail: str
