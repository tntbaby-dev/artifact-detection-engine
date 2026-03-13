# Audio Artifact Detection Engine

A comprehensive audio analysis toolkit for detecting quality issues in audio files.

## Features

- **Loudness Analysis**: LUFS measurement, segment-level loudness, inter/intra variance detection
- **Spectral Analysis**: Spectral balance scoring, sibilance detection (5-10kHz), spectral centroid tracking
- **Dynamics Analysis**: Dynamic range (crest factor), clipping detection
- **Noise Analysis**: Noise floor estimation, DC offset detection
- **Quality Scoring**: Weighted scoring system with configurable thresholds
- **Batch Processing**: Process multiple files with glob patterns
- **JSON Reports**: Detailed reports per file and batch summaries

## Installation

```bash
pip install -e .
```

Or with development dependencies:

```bash
pip install -e ".[dev]"
```

## Usage

### Command Line

Analyze a single file:

```bash
artifact-detect audio.wav
```

Analyze multiple files with glob patterns:

```bash
artifact-detect "audio/**/*.wav" -o reports/
```

Use a custom configuration:

```bash
artifact-detect audio.wav -c config.yaml
```

Get summary output:

```bash
artifact-detect *.wav --format summary
```

Filter by quality score:

```bash
artifact-detect audio/*.wav --min-score 80
```

### Python API

```python
from artifact_detection_engine import ArtifactDetectionEngine

# Initialize engine
engine = ArtifactDetectionEngine()

# Analyze a file
report = engine.analyze("audio.wav")

print(f"Quality Score: {report.quality_score}")
print(f"Duration: {report.duration}s")
print(f"Flags: {len(report.all_flags)}")

# Access detailed metrics
for result in report.analysis_results:
    print(f"{result.analyzer_name}: {result.metrics}")
```

### With Configuration

```python
config = {
    "analyzers": {
        "loudness": {"variance_threshold_lu": 2.0},
        "dynamics": {"clipping_threshold": 0.95},
    },
    "scoring": {
        "clipping_penalty": 25.0,
        "noise_floor_threshold": -35.0,
    }
}

engine = ArtifactDetectionEngine(config=config)
```

## Quality Scoring

Base score is 100 points. Deductions:

| Issue | Penalty | Cap |
|-------|---------|-----|
| Clipping | -20 per instance | -60 max |
| LUFS variance > 3 LU | -15 | - |
| Sibilance spikes | -10 per spike | -30 max |
| Noise floor > -40 dBFS | -15 | - |
| Dynamic range < 6 dB or > 40 dB | -10 | - |
| DC offset > 1% | -5 | - |

## Output Format

Each file report includes:

```json
{
  "file_path": "/path/to/audio.wav",
  "duration": 120.5,
  "sample_rate": 48000,
  "channels": 2,
  "quality_score": 85.0,
  "analysis_results": [...],
  "all_flags": [...],
  "summary": {
    "total_flags": 3,
    "flags_by_severity": {"warning": 2, "error": 1},
    "flags_by_type": {"clipping": 1, "sibilance": 2}
  }
}
```

## Configuration

See `config.example.yaml` for all available options.

## License

MIT
