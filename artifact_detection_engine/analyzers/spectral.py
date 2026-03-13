from __future__ import annotations

import numpy as np

from artifact_detection_engine.analyzers.base import AudioData
from artifact_detection_engine.models.results import AnalysisResult, SegmentFlag, Severity


class SpectralAnalyzer:
    name = "spectral"

    def analyze(self, audio: AudioData) -> AnalysisResult:
        x = audio.as_mono()
        sr = audio.sample_rate

        # Limit analysis to first 10 seconds for speed
        max_n = min(x.size, sr * 10)
        x = x[:max_n]

        n_fft = 4096
        hop = n_fft // 2

        if x.size < n_fft:
            return AnalysisResult(
                analyzer=self.name,
                metrics={"note": "audio too short for spectral analysis"},
                flags=[],
            )

        window = np.hanning(n_fft).astype(np.float32)
        mags = []

        for start in range(0, x.size - n_fft, hop):
            frame = x[start:start + n_fft] * window
            spec = np.fft.rfft(frame)
            mag = np.abs(spec)
            mags.append(mag)

        mean_mag = np.mean(np.stack(mags, axis=0), axis=0)
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

        def band_energy(f_lo: float, f_hi: float) -> float:
            idx = (freqs >= f_lo) & (freqs < f_hi)
            return float(np.sum(mean_mag[idx] ** 2) + 1e-12)

        total = band_energy(20, min(20000, sr / 2))

        # Skip heuristics if signal has almost no broadband energy
        if total <= 1e-6:
            return AnalysisResult(
                analyzer=self.name,
                metrics={"note": "signal energy too low for spectral heuristics"},
                flags=[],
            )

        presence = band_energy(2000, 5000)
        sibilance = band_energy(5000, 10000)

        presence_ratio = presence / total
        sibilance_ratio = sibilance / total

        flags = []

        if sibilance_ratio > 0.20:
            flags.append(
                SegmentFlag(
                    analyzer=self.name,
                    severity=Severity.info,
                    message=f"High sibilance-band energy ratio ({sibilance_ratio:.2f}). Could sound harsh.",
                    metrics={"sibilance_ratio": sibilance_ratio},
                )
            )

        if presence_ratio < 0.05 and sibilance_ratio > 0.01:
            flags.append(
                SegmentFlag(
                    analyzer=self.name,
                    severity=Severity.info,
                    message=f"Low presence-band energy ratio ({presence_ratio:.2f}). Could sound dull/muffled.",
                    metrics={"presence_ratio": presence_ratio},
                )
            )

        return AnalysisResult(
            analyzer=self.name,
            metrics={
                "presence_ratio": presence_ratio,
                "sibilance_ratio": sibilance_ratio,
            },
            flags=flags,
        )

