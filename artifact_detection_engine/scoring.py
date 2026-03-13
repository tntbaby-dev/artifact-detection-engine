from __future__ import annotations

from typing import List

from artifact_detection_engine.models.results import AnalysisResult, Severity


def compute_score(results: List[AnalysisResult]) -> float:
    score = 100.0

    for r in results:
        # Severity-based penalties
        for f in r.flags:
            if f.severity == Severity.critical:
                score -= 30
            elif f.severity == Severity.warn:
                score -= 15
            else:
                score -= 5

        # Analyzer-specific extra penalties (simple + effective)
        if r.analyzer == "dynamics":
            clip_ratio = float(r.metrics.get("clip_ratio", 0.0) or 0.0)
            # Scale penalty: 0% -> 0 points, 1% -> 10 points, 10% -> 50 points, 30%+ -> 80 points
            score -= min(80.0, clip_ratio * 500.0)
            
        if r.analyzer == "dropouts":
            dropout_count = int(r.metrics.get("dropout_count", 0) or 0)
            total_dropout_duration = float(r.metrics.get("total_dropout_duration_sec", 0.0) or 0.0)

            score -= min(25.0, dropout_count * 4.0 + total_dropout_duration * 5.0)

        if r.analyzer == "clicks":
            click_count = int(r.metrics.get("click_count", 0) or 0)
            score -= min(20.0, click_count * 2.5)

        if r.analyzer == "noise_bursts":
            burst_count = int(r.metrics.get("burst_count", 0) or 0)
            total_burst_duration = float(r.metrics.get("total_burst_duration_sec", 0.0) or 0.0)

            score -= min(20.0, burst_count * 3.0 + total_burst_duration * 8.0)

    return max(0.0, min(100.0, score))

