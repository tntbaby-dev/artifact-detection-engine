from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from artifact_detection_engine.models.results import FileReport


def report_to_dict(report: FileReport) -> Dict[str, Any]:
    return report.to_json_dict()


def write_json(report: FileReport, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = report_to_dict(report)
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path

