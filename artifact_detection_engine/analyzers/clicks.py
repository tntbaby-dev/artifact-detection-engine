from __future__ import annotations

import numpy as np

from artifact_detection_engine.analyzers.base import BaseAnalyzer
from artifact_detection_engine.models.results import AnalysisResult, SegmentFlag, Severity


class ClicksAnalyzer(BaseAnalyzer):
    name = "clicks"

    def __init__(
        self,
        frame_size: int = 1024,
        hop_size: int = 256,
        sensitivity: float = 20.0,
        min_separation_sec: float = 0.3,
        activity_threshold_db: float = -45.0,
    ):
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.sensitivity = sensitivity
        self.min_separation_sec = min_separation_sec
        self.activity_threshold_db = activity_threshold_db

    def _merge_flags(self, flags: list[SegmentFlag], merge_gap_sec: float = 0.5) -> list[SegmentFlag]:
        if not flags:
            return []

        merged = []
        current = flags[0]

        for nxt in flags[1:]:
            if nxt.start_sec - current.end_sec <= merge_gap_sec:
                current.end_sec = max(current.end_sec, nxt.end_sec)
                current.message = "Detected click/pop burst region"
                if nxt.severity == Severity.critical:
                    current.severity = Severity.critical
            else:
                merged.append(current)
                current = nxt

        merged.append(current)
        return merged
    
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
                    "click_count": 0,
                    "avg_flux": 0.0,
                    "threshold": 0.0,
                },
                flags=[],
            )

        window = np.hanning(self.frame_size)
        spectra = []
        frame_times = []
        frame_rms_db = []

        for start in range(0, len(mono) - self.frame_size + 1, self.hop_size):
            frame = mono[start:start + self.frame_size] 
            windowed = frame * window

            mag = np.abs(np.fft.rfft(windowed))
            spectra.append(mag)
            frame_times.append(start / audio.sample_rate)

            rms = float(np.sqrt(np.mean(frame ** 2)))
            rms_db = 20.0 * np.log10(max(rms, 1e-12))
            frame_rms_db.append(rms_db)

        spectra = np.asarray(spectra, dtype=np.float32)
        frame_rms_db = np.asarray(frame_rms_db)

        if len(spectra) < 2:
            return AnalysisResult(
                analyzer=self.name,
                metrics={
                    "click_count": 0,
                    "avg_flux": 0.0,
                    "threshold": 0.0,
                },
                flags=[],
            )

        # spectral flux: frame-to-frame positive spectral change
        flux = np.sum(np.maximum(0.0, spectra[1:] - spectra[:-1]), axis=1)

        median_flux = float(np.median(flux))
        threshold = median_flux * self.sensitivity

        candidate_indices = np.where(flux > threshold)[0]

        flags = []
        accepted_times = []

        min_gap_frames = int(self.min_separation_sec * audio.sample_rate / self.hop_size)

        last_idx = -min_gap_frames

        for idx in candidate_indices:

            # ignore transients in very quiet / silent regions
            if frame_rms_db[idx + 1] < self.activity_threshold_db:
                continue

            # peak picking: ensure local maximum
            if idx > 0 and idx < len(flux) - 1:
                if not (flux[idx] > flux[idx - 1] and flux[idx] > flux[idx + 1]):
                    continue

            # enforce minimum spacing
            if idx - last_idx < min_gap_frames:
                continue

            last_idx = idx

            event_time = frame_times[idx + 1]

            severity = Severity.warn
            if flux[idx] > threshold * 2.5:
                severity = Severity.critical

            flags.append(
                SegmentFlag(
                    analyzer=self.name,
                    severity=severity,
                    message="Detected click/pop transient",
                    start_sec=float(event_time),
                    end_sec=float(event_time + self.frame_size / audio.sample_rate),
                    metrics={
                        "spectral_flux": float(flux[idx]),
                        "threshold": float(threshold),
                    },
                )
            )
        flags = self._merge_flags(flags, merge_gap_sec=0.5)

        return AnalysisResult(
            analyzer=self.name,
            metrics={
                "click_count": len(flags),
                "avg_flux": float(np.mean(flux)) if len(flux) > 0 else 0.0,
                "median_flux": median_flux,
                "threshold": float(threshold),
                "activity_threshold_db": float(self.activity_threshold_db)
            },
            flags=flags,
        )