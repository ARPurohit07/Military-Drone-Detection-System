"""Stress-test a detector on an unlabeled deployment image directory.

This is deliberately separate from mAP evaluation: a directory of incident
frames is not a ground-truth dataset. It reports score distributions and must
be followed by human labeling before the images are added to training.

Example:
    python -m tools.benchmark_deployment_negatives --root detections --device cuda
"""

import argparse
import json
import time
from pathlib import Path

import cv2

from models.two_stage import TwoStageDroneDetector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="detections")
    parser.add_argument("--detector",
                        default="weights/binary_v5_birds_focused/best_model.pth")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--threshold", type=float, default=0.01)
    args = parser.parse_args()

    paths = sorted(p for p in Path(args.root).rglob("*")
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    detector_path = (args.detector if Path(args.detector).exists()
                     else "weights/best_model.pth")
    model = TwoStageDroneDetector(detector_path, classifier_path=None,
                                  device=args.device,
                                  score_thresh=args.threshold)
    start = time.perf_counter()
    rows = []
    for path in paths:
        image = cv2.imread(str(path))
        detections = model.predict(image) if image is not None else []
        score = max((float(d["score"]) for d in detections), default=0.0)
        rows.append({"image": str(path), "max_score": score})

    elapsed = time.perf_counter() - start
    scores = [r["max_score"] for r in rows]
    thresholds = (0.35, 0.50, 0.70, 0.90)
    report = {
        "root": args.root,
        "images": len(rows),
        "seconds": round(elapsed, 2),
        "fps": round(len(rows) / max(elapsed, 0.001), 2),
        "counts_at_threshold": {
            str(t): sum(s >= t for s in scores) for t in thresholds
        },
        "mean_max_score": round(sum(scores) / max(len(scores), 1), 4),
        "top_scores": sorted(rows, key=lambda r: r["max_score"], reverse=True)[:20],
        "warning": "Unlabeled stress-test results are not FPR or mAP.",
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
