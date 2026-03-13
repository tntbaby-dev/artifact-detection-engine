from __future__ import annotations

import numpy as np

from artifact_detection_engine.analyzers.base import BaseAnalyzer
from artifact_detection_engine.models.results import AnalysisResult, SegmentFlag, Severity


class DropoutAnalyzer(BaseAnalyzer):
    name = "dropouts"

    def __init__(
        self,
        frame_size: int = 2048,
        hop_size: int = 512,
        silence_threshold_db: float = -60.0,
        min_dropout_duration_sec: float = 0.2,
    ):
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.silence_threshold_db = silence_threshold_db
        self.min_dropout_duration_sec = min_dropout_duration_sec

    def analyze(self, audio) -> AnalysisResult:
        samples = audio.samples

        if samples.ndim == 2:
            mono = np.mean(samples, axis=1)
        else:
            mono = samples

        mono = mono.astype(np.float32)

        if len(mono) < self.frame_size:
            return AnalysisResult(
                analyzer=self.name,
                metrics={
                    "dropout_count": 0,
                    "total_dropout_duration_sec": 0.0,
                    "silence_threshold_db": self.silence_threshold_db,
                },
                flags=[],
            )

        frame_starts = range(0, len(mono) - self.frame_size + 1, self.hop_size)

        silent_frames = []
        frame_times = []

        for start in frame_starts:
            frame = mono[start:start + self.frame_size]
            rms = float(np.sqrt(np.mean(frame ** 2)))
            rms_db = 20.0 * np.log10(max(rms, 1e-12))

            silent_frames.append(rms_db < self.silence_threshold_db)
            frame_times.append(start / audio.sample_rate)

        silent_frames = np.array(silent_frames, dtype=bool)

        flags: list[SegmentFlag] = []
        total_dropout_duration = 0.0

        silent_indices = np.where(silent_frames)[0]

        if len(silent_indices) > 0:
            groups = []
            start_idx = silent_indices[0]
            prev_idx = silent_indices[0]

            for idx in silent_indices[1:]:
                if idx == prev_idx + 1:
                    prev_idx = idx
                else:
                    groups.append((start_idx, prev_idx))
                    start_idx = idx
                    prev_idx = idx

            groups.append((start_idx, prev_idx))

            for group_start, group_end in groups:
                start_sec = frame_times[group_start]
                end_sec = frame_times[group_end] + (self.frame_size / audio.sample_rate)
                duration_sec = end_sec - start_sec

                if duration_sec < self.min_dropout_duration_sec:
                    continue

                total_dropout_duration += duration_sec

                severity = Severity.warn
                if duration_sec >= 1.0:
                    severity = Severity.critical

                flags.append(
                    SegmentFlag(
                        analyzer=self.name,
                        severity=severity,
                        message=f"Detected dropout lasting {duration_sec:.3f} sec",
                        start_sec=float(start_sec),
                        end_sec=float(end_sec),
                        metrics={
                            "duration_sec": float(duration_sec),
                            "silence_threshold_db": float(self.silence_threshold_db),
                        },
                    )
                )

        return AnalysisResult(
            analyzer=self.name,
            metrics={
                "dropout_count": len(flags),
                "total_dropout_duration_sec": float(total_dropout_duration),
                "silence_threshold_db": float(self.silence_threshold_db),
            },
            flags=flags,
        )
