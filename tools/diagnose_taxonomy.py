"""Separate localization failure from country-attribution failure.

Example:
    python -m tools.diagnose_taxonomy \
      --root drone_dataset_country_v3 \
      --weights weights/country_v3/best_model.pth --split test \
      --out reports/country_v3_diagnosis.txt

Country mAP mixes two questions: whether the detector found the drone and
whether it assigned the right origin. They require different fixes, so this
tool reports them separately and lists the most common attribution confusions.
"""

import argparse
import collections
from pathlib import Path

import torch

from config import TrainConfig, load_label_map
from data.dataset import DroneDataset
from data.transforms import get_val_transforms
from models.faster_rcnn import build_model
from models.inference import load_weights, resolve_device
from utils.metrics import box_iou, collect_predictions, compute_mAP


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None)
    parser.add_argument("--weights", default="weights/country_v3/best_model.pth")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--score-thresh", type=float, default=0.05)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = TrainConfig()
    root = Path(args.root) if args.root else cfg.root
    class_map, scheme, source = load_label_map(root)
    class_names = {value: key for key, value in class_map.items()}
    device = resolve_device(args.device)

    lines = []

    def emit(message=""):
        print(message)
        lines.append(message)

    emit(f"Dataset: {root} ({scheme}, {len(class_map)} classes; {source})")
    emit(f"Checkpoint: {args.weights}")
    emit(f"Device: {device}")

    checkpoint_preview = torch.load(
        args.weights, map_location="cpu", weights_only=False)
    model_cfg = (checkpoint_preview.get("model_config", {})
                 if isinstance(checkpoint_preview, dict) else {})
    del checkpoint_preview
    model = build_model(
        len(class_map), pretrained=False,
        min_size=int(model_cfg.get("min_size", cfg.min_size)),
        max_size=int(model_cfg.get("max_size", cfg.max_size)))
    load_weights(model, args.weights, device,
                 expected_label_map=class_map, label_map_source=source)
    model.to(device).eval()

    dataset = DroneDataset(
        root, args.split, get_val_transforms(),
        skip_difficult=cfg.skip_difficult, min_score=cfg.min_score,
        max_area_frac=cfg.max_area_frac, label_map=class_map,
    ).positives_only()
    emit(f"{args.split.title()} images with drones: {len(dataset)}")
    predictions, ground_truths = collect_predictions(
        model, dataset, device, args.score_thresh)

    agnostic_predictions = [
        {"boxes": p["boxes"], "scores": p["scores"],
         "labels": torch.ones_like(p["labels"])} for p in predictions
    ]
    agnostic_ground_truths = [
        {"boxes": g["boxes"], "labels": torch.ones_like(g["labels"])}
        for g in ground_truths
    ]
    m50_agnostic, _ = compute_mAP(
        agnostic_predictions, agnostic_ground_truths, [1], iou_threshold=0.5)
    m75_agnostic, _ = compute_mAP(
        agnostic_predictions, agnostic_ground_truths, [1], iou_threshold=0.75)
    m50_country, _ = compute_mAP(
        predictions, ground_truths, sorted(class_map.values()),
        iou_threshold=0.5)

    emit("\n=== Localization versus attribution ===")
    emit(f"Class-agnostic mAP@0.5 : {m50_agnostic:.4f}")
    emit(f"Class-agnostic mAP@0.75: {m75_agnostic:.4f}")
    emit(f"Country-aware mAP@0.5  : {m50_country:.4f}")
    if m50_agnostic:
        cost = 100 * (1 - m50_country / m50_agnostic)
        emit(f"Relative mAP lost to attribution: {cost:.1f}%")

    hits = correct = 0
    topk = collections.Counter()
    confusion = collections.Counter()
    per_class_total = collections.Counter()
    per_class_right = collections.Counter()

    for prediction, ground_truth in zip(predictions, ground_truths):
        if not len(prediction["boxes"]) or not len(ground_truth["boxes"]):
            continue
        used_predictions = set()
        for ground_index, ground_box in enumerate(ground_truth["boxes"]):
            ious = box_iou(ground_box, prediction["boxes"])
            candidates = [
                int(i) for i in (ious >= 0.5).nonzero().flatten()
                if int(i) not in used_predictions
            ]
            if not candidates:
                continue
            candidates.sort(key=lambda i: float(prediction["scores"][i]),
                            reverse=True)
            best = candidates[0]
            used_predictions.add(best)
            true_label = int(ground_truth["labels"][ground_index])
            predicted_label = int(prediction["labels"][best])
            hits += 1
            per_class_total[true_label] += 1
            if predicted_label == true_label:
                correct += 1
                per_class_right[true_label] += 1
            else:
                confusion[(true_label, predicted_label)] += 1

            # Approximate ROI-head top-k from postprocessed overlapping boxes;
            # torchvision does not expose the full per-proposal class logits.
            label_scores = {}
            for index in candidates:
                label = int(prediction["labels"][index])
                label_scores[label] = max(
                    label_scores.get(label, 0.0),
                    float(prediction["scores"][index]))
            ranking = [label for label, _ in sorted(
                label_scores.items(), key=lambda item: -item[1])]
            for k in (1, 2, 3):
                if true_label in ranking[:k]:
                    topk[k] += 1

    emit("\n=== Country accuracy given IoU >= 0.5 ===")
    emit(f"Localized ground-truth boxes: {hits}")
    emit(f"Top-1 country correct: {correct}/{hits} "
         f"({100 * correct / max(hits, 1):.1f}%)")
    for k in (1, 2, 3):
        emit(f"Top-{k} country accuracy: {topk[k]}/{hits} "
             f"({100 * topk[k] / max(hits, 1):.1f}%)")
    emit(f"Uniform-chance top-1 baseline: "
         f"{100 / max(len(class_map), 1):.1f}%")

    emit("\nPer-country accuracy among localized boxes:")
    for class_id in sorted(
            per_class_total,
            key=lambda c: (-per_class_total[c], class_names[c])):
        total = per_class_total[class_id]
        right = per_class_right[class_id]
        emit(f"  {class_names[class_id]:<18} {right:4d}/{total:<4d} "
             f"{100 * right / total:6.1f}%")

    emit("\nMost common country confusions:")
    if not confusion:
        emit("  none")
    for (true_label, predicted_label), count in confusion.most_common(20):
        emit(f"  {count:4d}  {class_names[true_label]:<18} -> "
             f"{class_names[predicted_label]}")

    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nReport written to {output}")


if __name__ == "__main__":
    main()
