from __future__ import annotations

import csv
import json
from pathlib import Path

from artifact_detection_engine.engine import ArtifactDetectionEngine


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".aiff", ".ogg"}


def summarize_report(report) -> dict:
    summary = {
        "file_path": report.file_path,
        "score": float(report.score),
        "sample_rate": int(report.sample_rate),
        "duration_sec": float(report.duration_sec),
    }

    for result in report.results:
        analyzer = result.analyzer

        if analyzer == "clipping":
            summary["clipping_flags"] = len(result.flags)
            summary["clip_ratio"] = float(result.metrics.get("clip_ratio", 0.0) or 0.0)

        elif analyzer == "dropouts":
            summary["dropout_flags"] = len(result.flags)
            summary["total_dropout_duration_sec"] = float(
                result.metrics.get("total_dropout_duration_sec", 0.0) or 0.0
            )

        elif analyzer == "clicks":
            summary["click_flags"] = len(result.flags)
            summary["click_count"] = int(result.metrics.get("click_count", 0) or 0)

        elif analyzer == "noise_bursts":
            summary["noise_burst_flags"] = len(result.flags)
            summary["burst_count"] = int(result.metrics.get("burst_count", 0) or 0)

    return summary


def find_audio_files(folder: Path) -> list[Path]:
    files = []
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            files.append(path)
    return sorted(files)


def main() -> None:
    target_folder = Path("/Users/mac/Desktop/Python Projects/loudness_engine/cycles_vox")
    output_folder = Path("reports")
    output_folder.mkdir(exist_ok=True)

    engine = ArtifactDetectionEngine()
    audio_files = find_audio_files(target_folder)

    if not audio_files:
        print("No audio files found.")
        return

    print(f"Found {len(audio_files)} audio files.")

    summaries = []
    full_reports = []

    for i, audio_file in enumerate(audio_files, start=1):
        print(f"[{i}/{len(audio_files)}] Analyzing: {audio_file.name}")

        try:
            report = engine.analyze_file(audio_file)
            summaries.append(summarize_report(report))
            full_reports.append(report.model_dump(mode="json"))
            print(f"  Done | score={report.score}")
        except Exception as e:
            print(f"  Failed | {audio_file} | {e}")

    summaries.sort(key=lambda x: x["score"])
    full_reports.sort(key=lambda x: x["score"])

    json_path = output_folder / "batch_report.json"
    csv_path = output_folder / "batch_report.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_reports, f, indent=2)

    if summaries:
        fieldnames = sorted({key for row in summaries for key in row.keys()})
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summaries)

    print("\nBatch analysis complete.")
    print(f"JSON report: {json_path}")
    print(f"CSV report:  {csv_path}")

    worst = summaries[:5]
    if worst:
        print("\nWorst files:")
        for row in worst:
            print(f" - score={row['score']:.1f} | {row['file_path']}")


if __name__ == "__main__":
    main()