from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Severity(str, Enum):
    info = "info"
    warn = "warn"
    critical = "critical"


class SegmentFlag(BaseModel):
    analyzer: str = Field(..., description="Analyzer name that produced this flag")
    severity: Severity
    message: str

    start_sec: Optional[float] = Field(None, ge=0)
    end_sec: Optional[float] = Field(None, ge=0)

    metrics: Dict[str, Any] = Field(default_factory=dict)


class AnalysisResult(BaseModel):
    analyzer: str = Field(..., description="Analyzer name")
    metrics: Dict[str, Any] = Field(default_factory=dict)
    flags: List[SegmentFlag] = Field(default_factory=list)


class AudioData(BaseModel):
    samples: Any
    sample_rate: int = Field(..., gt=0)
    duration_sec: float = Field(..., ge=0)

    
class FileReport(BaseModel):
    file_path: str
    sample_rate: int = Field(..., gt=0)
    duration_sec: float = Field(..., ge=0)

    results: List[AnalysisResult] = Field(default_factory=list)
    score: float = Field(..., ge=0, le=100)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_json_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")
