# Model Card
Owner: Muteb · fill during Phase 2

> Fill in at the end of Phase 2. Every number states split, input resolution and thresholds.

## Model details
- Detector (Ali): YOLO (version, checkpoint `weights/best.pt`, SHA1 from `manifest.json`)
- Team classifier (Faisal): variant ___, checkpoint `runs/classify/___/best.pt`
- Pitch segmentation (Fahd): YOLO11n-seg (`weights/pitch_seg.pt`); homography from marking intersections + RANSAC, rejected when the markings do not fit the pitch model
- Input resolution: 1280; confidence threshold: ___ (from threshold sweep); NMS IoU: ___
- Training data: SoccerNet-Tracking train split (games: see `data/yolo/split.json`)

## Intended use
- Users: performance analysts.
- Decision supported: positional statistics such as team shape and spacing from top-down player positions.

## Out-of-scope uses
- Identifying people outside the pitch (spectators, staff).
- Footage not covered by the dataset licence.
- Any use as a refereeing decision tool.

## Acceptance criteria (written BEFORE results, date: 2026-10-11)
| Criterion | Target | Result | Pass? |
|---|---|---|---|
| Player AP50 (test) | ≥ 0.85 | | |
| Ball recall at operating point | ≥ 0.50 | | |
| Team-assignment accuracy (test) | ≥ 0.90 | | |
| Pitch-marking IoU (test) | ≥ 0.60 | | |
| Homography reprojection error | ≤ 2 m | | |
| End-to-end throughput (T4, 1280) | ≥ 10 FPS | | |
| p95 detector latency (T4, 1280, batch 1) | ≤ 200 ms | | |

## Evaluation
- Detection: mAP50, mAP50–95, per-class AP, recall by object size.
- Classification: accuracy, per-class F1, ECE (team assignment).
- Segmentation: mask mAP50–95, mean IoU, Dice; homography reprojection error.
- Benchmark: `runs/benchmark.json` (warm-up, iterations, device stated).

## Failure analysis
Link the failure gallery; one sentence per failure mode (e.g. ball lost under motion blur; goalkeeper labelled as referee; close-up breaks the homography).

## Limitations
- 12 games only; broadcast camera only.
- Shot changes and close-ups invalidate the homography (detected or reported).

## Go / No-Go decision
State the decision and which criteria drove it.
