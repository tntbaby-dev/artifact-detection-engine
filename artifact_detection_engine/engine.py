from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf

from artifact_detection_engine.analyzers.noise_bursts import NoiseBurstsAnalyzer
from artifact_detection_engine.analyzers.clicks import ClicksAnalyzer
from artifact_detection_engine.analyzers.dropouts import DropoutAnalyzer
from artifact_detection_engine.analyzers.base import BaseAnalyzer
from artifact_detection_engine.analyzers.clipping import ClippingAnalyzer
from artifact_detection_engine.models.results import AudioData, FileReport
from artifact_detection_engine.scoring import compute_score


class ArtifactDetectionEngine:
    def __init__(self, analyzers: List[BaseAnalyzer] | None = None):
        if analyzers is None:
            analyzers = [
                ClippingAnalyzer(),
                DropoutAnalyzer(),
                ClicksAnalyzer(),
                NoiseBurstsAnalyzer(),
            ]
        self.analyzers = analyzers

    def load_audio(self, file_path: str | Path) -> AudioData:
        path = Path(file_path)
        samples, sr = sf.read(str(path), always_2d=True)
        x = np.asarray(samples, dtype=np.float32)

        return AudioData(
            samples=x,
            sample_rate=int(sr),
            duration_sec=float(len(x) / sr),
        )

    def analyze_file(self, file_path):

        audio = self.load_audio(file_path)

        results = []

        for analyzer in self.analyzers:
            result = analyzer.analyze(audio)

            if result is None:
                continue

            results.append(result)

        score = compute_score(results)

        report = FileReport(
            file_path=str(file_path),
            sample_rate=audio.sample_rate,
            duration_sec=audio.duration_sec,
            results=results,
            score=score,
        )

        return report