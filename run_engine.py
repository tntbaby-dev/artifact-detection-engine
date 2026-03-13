from artifact_detection_engine.engine import ArtifactDetectionEngine

engine = ArtifactDetectionEngine()

report = engine.analyze_file(
    "/Users/mac/Desktop/Python Projects/loudness_engine/cycles_vox/chorus_vox.wav"
)

for result in report.results:
    print(f"\nAnalyzer: {result.analyzer}")
    print("Metrics:", result.metrics)
    print("Flags:", len(result.flags))

    for flag in result.flags[:10]:
        print(
            f" - {flag.severity.value} | "
            f"{flag.start_sec:.3f}s -> {flag.end_sec:.3f}s | "
            f"{flag.message}"
        )

print(f"\nFinal Score: {report.score}")