"""Run the two-Faster-R-CNN detector and attribution pipeline.

Examples:
    python predict.py --image view1.jpg
    python predict.py --image view1.jpg --output runs/view1.jpg --json runs/view1.json
"""

import argparse
import json
from pathlib import Path

import cv2

from models.inference import resolve_device
from models.two_stage import TwoStageDroneDetector
from utils.visualization import draw_boxes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument(
        "--detector",
        default="weights/binary_v5_birds_focused/best_model.pth",
    )
    parser.add_argument("--country-detector",
                        default="weights/country_v3/best_model.pth")
    parser.add_argument("--classifier", default="weights/crop_country.pth",
                        help="legacy fallback when no country detector exists")
    parser.add_argument("--output", help="Optional annotated image path")
    parser.add_argument("--json", dest="json_path", help="Optional result JSON")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--threshold", type=float, default=0.50,
        help="Binary score threshold (default: recall-oriented demo point 0.50)",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"Could not read image: {image_path}")

    device = resolve_device(None if args.device == "auto" else args.device)
    detector_path = (args.detector if Path(args.detector).exists()
                     else "weights/best_model.pth")
    country_path = (args.country_detector
                    if Path(args.country_detector).exists() else None)
    model = TwoStageDroneDetector(
        detector_path, classifier_path=args.classifier,
        country_detector_path=country_path, device=device,
        score_thresh=args.threshold)
    results = model.predict(image)

    print(json.dumps({"image": str(image_path), "device": str(device),
                      "pipeline": model.describe(),
                      "detections": results}, indent=2))

    if args.json_path:
        path = Path(args.json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        shown = []
        for result in results:
            item = dict(result)
            # The rendered score belongs to the binary detector. Country
            # candidates remain available in JSON and must not be presented as
            # if this score were calibrated attribution confidence.
            item["label"] = "Drone"
            shown.append(item)
        cv2.imwrite(str(path), draw_boxes(image, shown))
        print(f"Annotated image written to {path}")


if __name__ == "__main__":
    main()
