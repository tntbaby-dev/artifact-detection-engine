from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich import print as rich_print

from artifact_detection_engine.engine import ArtifactDetectionEngine
from artifact_detection_engine.analyzers.noise import NoiseAnalyzer
from artifact_detection_engine.analyzers.dynamics import DynamicsAnalyzer
from artifact_detection_engine.analyzers.spectral import SpectralAnalyzer
from artifact_detection_engine.reporter import write_json
from artifact_detection_engine.analyzers.loudness import LoudnessAnalyzer
app = typer.Typer(add_completion=False)


def build_engine() -> ArtifactDetectionEngine:
    return ArtifactDetectionEngine([
    NoiseAnalyzer(),
    LoudnessAnalyzer(),
    DynamicsAnalyzer(),
    SpectralAnalyzer(),
])


@app.command()
def analyze(
    file: Path = typer.Argument(..., exists=True, readable=True),
    out: Optional[Path] = typer.Option(None, "--out", "-o"),
):
    engine = build_engine()
    report = engine.analyze_file(file)

    if out is None:
        out = file.with_suffix(".artifact_report.json")

    write_json(report, out)

    rich_print(f"[green]Done[/green] -> {out}")
    rich_print(f"Score: [bold]{report.score:.1f}[/bold]")

    any_flags = any(r.flags for r in report.results)
    if any_flags:
        print("Flags:")
        for r in report.results:
            for flag in r.flags:
                print(f" - [{r.analyzer} | {flag.severity.value}] {flag.message}")


@app.command()
def batch(
    folder: Path = typer.Argument(..., exists=True, readable=True),
    out_dir: Path = typer.Option(Path("reports"), "--out-dir"),
    pattern: str = typer.Option("*.wav", "--pattern"),
):
    engine = build_engine()
    files = sorted(folder.glob(pattern))

    if not files:
        rich_print(f"[yellow]No files matched[/yellow] {pattern} in {folder}")
        raise typer.Exit(code=1)

    out_dir.mkdir(parents=True, exist_ok=True)

    failed = 0

    for f in files:
        try:
            report = engine.analyze_file(f)
            out_path = out_dir / f"{f.stem}.artifact_report.json"
            write_json(report, out_path)
            print(f"{f.name} -> score {report.score:.1f} -> {out_path}")
        except Exception as e:
            failed += 1
            rich_print(f"[red]FAILED[/red] {f.name}: {e}")

    rich_print(
        f"[green]Batch complete[/green]. "
        f"Wrote {len(files) - failed} reports to {out_dir}. "
        f"Failed: {failed}"
    )


if __name__ == "__main__":
    app()
    
