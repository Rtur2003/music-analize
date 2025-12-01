from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class GenreScore(BaseModel):
    label: str = Field(..., description="Predicted genre label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Probability score")


class AnalysisResponse(BaseModel):
    filename: str
    genre: Optional[List[GenreScore]] = None
    authenticity_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    features: Dict[str, float]
    report_path: str
    message: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
