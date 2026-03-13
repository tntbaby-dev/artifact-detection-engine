from __future__ import annotations

import numpy as np

from artifact_detection_engine.analyzers.base import AudioData
from artifact_detection_engine.models.results import AnalysisResult, SegmentFlag, Severity


class NoiseAnalyzer:
    name = "noise"

    def analyze(self, audio: AudioData) -> AnalysisResult:
        x = audio.as_mono()

        # DC offset
        dc_offset = float(np.mean(x))

        # RMS in dBFS
        rms = float(np.sqrt(np.mean(x**2)) + 1e-12)
        rms_dbfs = float(20.0 * np.log10(rms))

        flags = []

        if abs(dc_offset) > 0.01:
            flags.append(
                SegmentFlag(
                    analyzer=self.name,
                    severity=Severity.warn,
                    message="DC offset is higher than expected.",
                    metrics={"dc_offset": dc_offset},
                )
            )

        if rms_dbfs > -25.0:
            flags.append(
                SegmentFlag(
                    analyzer=self.name,
                    severity=Severity.info,
                    message=f"Overall RMS is fairly high: {rms_dbfs:.1f} dBFS.",
                    metrics={"rms_dbfs": rms_dbfs},
                )
            )

        return AnalysisResult(
            analyzer=self.name,
            metrics={
                "dc_offset": dc_offset,
                "rms_dbfs": rms_dbfs,
            },
            flags=flags,
        )

