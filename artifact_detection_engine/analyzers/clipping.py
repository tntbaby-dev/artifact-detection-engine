from __future__ import annotations

import numpy as np

from artifact_detection_engine.analyzers.base import BaseAnalyzer
from artifact_detection_engine.models.results import AnalysisResult, SegmentFlag, Severity


class ClippingAnalyzer(BaseAnalyzer):
    name = "clipping"

    def __init__(self, threshold: float = 0.98, min_consecutive_samples: int = 3):
        self.threshold = threshold
        self.min_consecutive_samples = min_consecutive_samples

    def analyze(self, audio) -> AnalysisResult:
        samples = audio.samples

        # Convert to mono for simpler event detection
        if samples.ndim == 2:
            mono = np.mean(samples, axis=1)
        else:
            mono = samples

        mono = mono.astype(np.float32)

        clipped = np.abs(mono) >= self.threshold
        clip_indices = np.where(clipped)[0]

        flags: list[SegmentFlag] = []

        if len(clip_indices) > 0:
            groups = []
            start = clip_indices[0]
            prev = clip_indices[0]

            for idx in clip_indices[1:]:
                if idx == prev + 1:
                    prev = idx
                else:
                    groups.append((start, prev))
                    start = idx
                    prev = idx

            groups.append((start, prev))

            for start_idx, end_idx in groups:
                run_length = end_idx - start_idx + 1

                if run_length < self.min_consecutive_samples:
                    continue

                start_sec = start_idx / audio.sample_rate
                end_sec = end_idx / audio.sample_rate

                severity = Severity.warn
                if run_length >= 20:
                    severity = Severity.critical

                flags.append(
                    SegmentFlag(
                        analyzer=self.name,
                        severity=severity,
                        message=f"Detected clipping run of {run_length} samples",
                        start_sec=float(start_sec),
                        end_sec=float(end_sec),
                        metrics={
                            "run_length_samples": int(run_length),
                            "peak_abs": float(np.max(np.abs(mono[start_idx:end_idx + 1]))),
                        },
                    )
                )

        clip_ratio = float(np.mean(clipped)) if len(mono) > 0 else 0.0
        max_peak = float(np.max(np.abs(mono))) if len(mono) > 0 else 0.0

        return AnalysisResult(
            analyzer=self.name,
            metrics={
                "clip_ratio": clip_ratio,
                "max_abs_peak": max_peak,
                "num_clip_flags": len(flags),
                "threshold": self.threshold,
            },
            flags=flags,
        )