from __future__ import annotations

import numpy as np

from artifact_detection_engine.analyzers.base import BaseAnalyzer
from artifact_detection_engine.models.results import AnalysisResult, SegmentFlag, Severity


class NoiseBurstsAnalyzer(BaseAnalyzer):
    name = "noise_bursts"

    def __init__(
        self,
        frame_size: int = 2048,
        hop_size: int = 512,
        activity_threshold_db: float = -45.0,
        hf_cutoff_hz: float = 1000.0,
        sensitivity: float = 4.0,
        merge_gap_sec: float = 0.20,
        min_burst_duration_sec: float = 0.05,
    ):
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.activity_threshold_db = activity_threshold_db
        self.hf_cutoff_hz = hf_cutoff_hz
        self.sensitivity = sensitivity
        self.merge_gap_sec = merge_gap_sec
        self.min_burst_duration_sec = min_burst_duration_sec

    def analyze(self, audio) -> AnalysisResult:
        samples = audio.samples

        if samples.ndim == 2:
            mono = np.mean(samples, axis=1)
        else:
            mono = samples

        mono = mono.astype(np.float32)

        if len(mono) < self.frame_size * 2:
            return AnalysisResult(
                analyzer=self.name,
                metrics={
                    "burst_count": 0,
                    "avg_burst_score": 0.0,
                    "activity_threshold_db": float(self.activity_threshold_db),
                    "hf_cutoff_hz": float(self.hf_cutoff_hz),
                },
                flags=[],
            )

        sr = audio.sample_rate
        window = np.hanning(self.frame_size)

        hf_energy = []
        frame_times = []
        frame_rms_db = []

        freqs = np.fft.rfftfreq(self.frame_size, d=1.0 / sr)
        hf_mask = freqs >= self.hf_cutoff_hz

        for start in range(0, len(mono) - self.frame_size + 1, self.hop_size):
            frame = mono[start:start + self.frame_size]
            windowed = frame * window

            mag = np.abs(np.fft.rfft(windowed))
            energy = float(np.sum(mag[hf_mask]))
            hf_energy.append(energy)
            frame_times.append(start / sr)

            rms = float(np.sqrt(np.mean(frame ** 2)))
            rms_db = 20.0 * np.log10(max(rms, 1e-12))
            frame_rms_db.append(rms_db)

        hf_energy = np.asarray(hf_energy, dtype=np.float32)
        frame_rms_db = np.asarray(frame_rms_db, dtype=np.float32)

        if len(hf_energy) < 5:
            return AnalysisResult(
                analyzer=self.name,
                metrics={
                    "burst_count": 0,
                    "avg_burst_score": 0.0,
                    "activity_threshold_db": float(self.activity_threshold_db),
                    "hf_cutoff_hz": float(self.hf_cutoff_hz),
                },
                flags=[],
            )

        burst_scores = np.zeros_like(hf_energy, dtype=np.float32)

        context = 4
        eps = 1e-8

        for i in range(len(hf_energy)):
            left = max(0, i - context)
            right = min(len(hf_energy), i + context + 1)

            neighborhood = hf_energy[left:right]

            if len(neighborhood) <= 1:
                burst_scores[i] = 1.0
                continue

            baseline = float(np.median(neighborhood))
            burst_scores[i] = float(hf_energy[i] / max(baseline, eps))

        candidate_indices = np.where(
            (burst_scores > self.sensitivity)
            & (frame_rms_db > self.activity_threshold_db)
        )[0]

        raw_flags: list[SegmentFlag] = []

        for idx in candidate_indices:
            if idx > 0 and idx < len(burst_scores) - 1:
                if not (
                    burst_scores[idx] > burst_scores[idx - 1]
                    and burst_scores[idx] >= burst_scores[idx + 1]
                ):
                    continue

            start_sec = float(frame_times[idx])
            end_sec = float(start_sec + self.frame_size / sr)
            score = float(burst_scores[idx])

            severity = Severity.warn
            if score > self.sensitivity * 2.0:
                severity = Severity.critical

            raw_flags.append(
                SegmentFlag(
                    analyzer=self.name,
                    severity=severity,
                    message="Detected noise burst region",
                    start_sec=start_sec,
                    end_sec=end_sec,
                    metrics={
                        "burst_score": score,
                        "hf_energy": float(hf_energy[idx]),
                        "frame_rms_db": float(frame_rms_db[idx]),
                    },
                )
            )

        flags = self._merge_flags(raw_flags)

        burst_durations = [
            float(flag.end_sec - flag.start_sec)
            for flag in flags
            if flag.start_sec is not None and flag.end_sec is not None
        ]

        return AnalysisResult(
            analyzer=self.name,
            metrics={
                "burst_count": len(flags),
                "avg_burst_score": float(np.mean(burst_scores)) if len(burst_scores) > 0 else 0.0,
                "max_burst_score": float(np.max(burst_scores)) if len(burst_scores) > 0 else 0.0,
                "total_burst_duration_sec": float(sum(burst_durations)),
                "activity_threshold_db": float(self.activity_threshold_db),
                "hf_cutoff_hz": float(self.hf_cutoff_hz),
                "sensitivity": float(self.sensitivity),
            },
            flags=flags,
        )

    def _merge_flags(self, flags: list[SegmentFlag]) -> list[SegmentFlag]:
        if not flags:
            return []

        flags = sorted(flags, key=lambda f: f.start_sec or 0.0)

        merged: list[SegmentFlag] = []
        current = flags[0]

        for nxt in flags[1:]:
            if current.end_sec is None or nxt.start_sec is None or nxt.end_sec is None:
                continue

            if nxt.start_sec - current.end_sec <= self.merge_gap_sec:
                current.end_sec = max(current.end_sec, nxt.end_sec)
                current.message = "Detected noise burst region"

                current_score = float(current.metrics.get("burst_score", 0.0))
                next_score = float(nxt.metrics.get("burst_score", 0.0))
                current.metrics["burst_score"] = max(current_score, next_score)

                if nxt.severity == Severity.critical:
                    current.severity = Severity.critical
            else:
                duration = float((current.end_sec or 0.0) - (current.start_sec or 0.0))
                if duration >= self.min_burst_duration_sec:
                    merged.append(current)
                current = nxt

        duration = float((current.end_sec or 0.0) - (current.start_sec or 0.0))
        if duration >= self.min_burst_duration_sec:
            merged.append(current)

        return merged