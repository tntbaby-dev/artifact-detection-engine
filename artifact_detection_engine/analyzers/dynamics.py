from __future__ import annotations

import numpy as np

from artifact_detection_engine.analyzers.base import AudioData
from artifact_detection_engine.models.results import AnalysisResult, SegmentFlag, Severity


class DynamicsAnalyzer:
    name = "dynamics"

    def analyze(self, audio: AudioData) -> AnalysisResult:
        x = audio.as_mono()

        # Basic levels
        peak = float(np.max(np.abs(x)) + 1e-12)
        peak_dbfs = float(20.0 * np.log10(peak))

        rms = float(np.sqrt(np.mean(x**2)) + 1e-12)
        rms_dbfs = float(20.0 * np.log10(rms))

        # Crest factor (dynamic-ish): peak - rms
        crest_db = float(peak_dbfs - rms_dbfs)

        # Clipping detection
        clip_threshold = 0.999  # assumes normalized float audio [-1, 1]
        clipped = np.abs(x) >= clip_threshold
        clip_count = int(np.sum(clipped))
        clip_ratio = float(clip_count / max(1, x.size))

        flags = []

        if clip_count > 0:
            severity = Severity.critical if clip_ratio > 0.001 else Severity.warn
            flags.append(
                SegmentFlag(
                    analyzer=self.name,
                    severity=severity,
                    message=f"Clipping detected: {clip_count} samples ({clip_ratio*100:.4f}%).",
                    metrics={"clip_count": clip_count, "clip_ratio": clip_ratio},
                )
            )

        # Very rough “over-compressed” heuristic
        if crest_db < 6.0 and rms_dbfs > -25.0:
            flags.append(
                SegmentFlag(
                    analyzer=self.name,
                    severity=Severity.info,
                    message=f"Low crest factor ({crest_db:.1f} dB) with high RMS ({rms_dbfs:.1f} dBFS): may be over-compressed.",
                    metrics={"crest_db": crest_db, "rms_dbfs": rms_dbfs},
                )
            )

        return AnalysisResult(
            analyzer=self.name,
            metrics={
                "peak_dbfs": peak_dbfs,
                "rms_dbfs": rms_dbfs,
                "crest_db": crest_db,
                "clip_count": clip_count,
                "clip_ratio": clip_ratio,
            },
            flags=flags,
        )

