from __future__ import annotations

import numpy as np

from artifact_detection_engine.analyzers.base import AudioData
from artifact_detection_engine.models.results import AnalysisResult, SegmentFlag, Severity


class LoudnessAnalyzer:
    name = "loudness"

    def __init__(
        self,
        window_sec: float = 0.4,
        hop_sec: float = 0.1,
        too_loud_rms_dbfs: float = -12.0,
        too_quiet_rms_dbfs: float = -35.0,
        swing_db: float = 18.0,
    ):
        self.window_sec = float(window_sec)
        self.hop_sec = float(hop_sec)
        self.too_loud_rms_dbfs = float(too_loud_rms_dbfs)
        self.too_quiet_rms_dbfs = float(too_quiet_rms_dbfs)
        self.swing_db = float(swing_db)

    @staticmethod
    def _rms_dbfs(x: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(x**2)) + 1e-12)
        return float(20.0 * np.log10(rms))

    def analyze(self, audio: AudioData) -> AnalysisResult:
        x = audio.as_mono()
        sr = audio.sample_rate

        integrated_rms_dbfs = self._rms_dbfs(x)

        win = max(1, int(self.window_sec * sr))
        hop = max(1, int(self.hop_sec * sr))

        short_terms = []
        times = []

        if x.size >= win:
            for start in range(0, x.size - win + 1, hop):
                frame = x[start:start + win]
                short_terms.append(self._rms_dbfs(frame))
                times.append(start / sr)

        flags = []

        if integrated_rms_dbfs > self.too_loud_rms_dbfs:
            flags.append(
                SegmentFlag(
                    analyzer=self.name,
                    severity=Severity.warn,
                    message=f"Integrated RMS is loud ({integrated_rms_dbfs:.1f} dBFS).",
                    metrics={"integrated_rms_dbfs": integrated_rms_dbfs},
                )
            )

        if integrated_rms_dbfs < self.too_quiet_rms_dbfs:
            flags.append(
                SegmentFlag(
                    analyzer=self.name,
                    severity=Severity.warn,
                    message=f"Integrated RMS is quiet ({integrated_rms_dbfs:.1f} dBFS).",
                    metrics={"integrated_rms_dbfs": integrated_rms_dbfs},
                )
            )

        if short_terms:
            st = np.array(short_terms, dtype=np.float32)
            st_min = float(st.min())
            st_max = float(st.max())
            st_range = float(st_max - st_min)

            if st_range > self.swing_db:
                idx_max = int(st.argmax())
                t_max = float(times[idx_max])
                flags.append(
                    SegmentFlag(
                        analyzer=self.name,
                        severity=Severity.info,
                        message=f"Large loudness swing (short-term range {st_range:.1f} dB).",
                        start_sec=t_max,
                        end_sec=min(audio.duration_sec, t_max + self.window_sec),
                        metrics={
                            "short_term_min_dbfs": st_min,
                            "short_term_max_dbfs": st_max,
                            "short_term_range_db": st_range,
                        },
                    )
                )

            metrics = {
                "integrated_rms_dbfs": integrated_rms_dbfs,
                "short_term_min_dbfs": st_min,
                "short_term_max_dbfs": st_max,
                "short_term_range_db": st_range,
                "window_sec": self.window_sec,
                "hop_sec": self.hop_sec,
            }
        else:
            metrics = {
                "integrated_rms_dbfs": integrated_rms_dbfs,
                "note": "audio too short for short-term loudness",
                "window_sec": self.window_sec,
                "hop_sec": self.hop_sec,
            }

        return AnalysisResult(
            analyzer=self.name,
            metrics=metrics,
            flags=flags,
        )
