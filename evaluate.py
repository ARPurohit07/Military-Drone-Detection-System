"""Evaluate a checkpoint: mAP, per-class AP, false-positive rate, and robustness.

The adverse-condition report is the part that is new. "Can we still detect it in
bad conditions" was one of the project's three original goals, but nothing
measured it - so the answer was unknown rather than bad. Passing
`--adverse` re-runs the same validation images under each named degradation from
`data.transforms.ADVERSE_SUITE` and prints mAP per condition next to the clear
baseline, which turns the goal into a number that can regress.

AP is pooled across the dataset per class (VOC-style), not averaged per image.
"""

import argparse
import math
import os
import time
from pathlib import Path

import torch

from config import (CLASS_MAP, CLASS_NAMES, LABEL_SCHEME, TrainConfig,
                    load_label_map)
from data.dataset import DroneDataset
from data.transforms import ADVERSE_SUITE, get_val_transforms
from models.faster_rcnn import build_model
from models.inference import load_weights, resolve_device
from utils.metrics import (
    collect_predictions,
    compute_mAP,
    compute_mAP_range,
)


def wilson_interval(successes, total, z=1.96):
    """95% Wilson score interval for a binomial rate."""
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = (z / denominator) * math.sqrt(
        p * (1 - p) / total + z * z / (4 * total * total))
    return center - radius, center + radius


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=str, default=None,
                        help="checkpoint (defaults to the current model for the selected mode)")
    parser.add_argument("--root", type=str, default=None,
                        help="dataset root (defaults to TrainConfig.root)")
    parser.add_argument("--score-thresh", type=float, default=0.05)
    parser.add_argument("--fpr-thresh", type=float, default=0.5)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N images (smoke test).")
    parser.add_argument("--adverse", action="store_true",
                        help="also report mAP under each adverse condition")
    parser.add_argument("--conditions", nargs="*", default=None,
                        help="restrict --adverse to these condition names")
    parser.add_argument("--include-difficult", action="store_true",
                        help="count low-confidence auto-labelled boxes as GT")
    parser.add_argument("--out", type=str, default=None,
                        help="Also write this report to a file.")
    parser.add_argument("--binary", action="store_true",
                        help="evaluate a single-class drone detector")
    parser.add_argument("--split", choices=["val", "test"], default="val",
                        help="held-out split to evaluate")
    parser.add_argument("--cache", type=str, default=None,
                        help="resumable CPU prediction cache for long GPU runs")
    args = parser.parse_args()

    lines = []

    def emit(text=""):
        print(text)
        lines.append(text)

    cfg = TrainConfig()
    root = Path(args.root) if args.root else (
        Path("drone_dataset_binary_v4_birds") if args.binary else cfg.root)
    weights = args.weights or (
        "weights/binary_v5_birds_focused/best_model.pth" if args.binary
        else "weights/country_v3/best_model.pth")
    device = resolve_device(args.device)
    emit(f"Using device: {device}")
    loaded_map, loaded_scheme, _ = load_label_map(root)
    active_map = {"drone": 1} if args.binary else loaded_map
    active_names = {v: k for k, v in active_map.items()}
    active_scheme = "drone (binary)" if args.binary else loaded_scheme
    emit(f"Dataset: {root}  (label scheme: {active_scheme})")

    checkpoint_preview = torch.load(
        weights, map_location="cpu", weights_only=False)
    saved_model_cfg = (checkpoint_preview.get("model_config", {})
                       if isinstance(checkpoint_preview, dict) else {})
    del checkpoint_preview
    model_min_size = int(saved_model_cfg.get("min_size", cfg.min_size))
    model_max_size = int(saved_model_cfg.get("max_size", cfg.max_size))
    model = build_model(
        len(active_map), pretrained=False,
        min_size=model_min_size, max_size=model_max_size,
    )
    meta = load_weights(
        model, weights, device, strict_labels=not args.binary,
        expected_label_map=active_map, label_map_source=str(root / "classes.json"))
    if meta.get("epoch"):
        emit(f"Checkpoint: {weights} (epoch {meta['epoch']}, "
             f"val_loss {meta.get('val_loss', float('nan')):.4f})")
    model.to(device)
    model.eval()
    emit(f"Model input resize: short side {model_min_size}, "
         f"long side <= {model_max_size}")

    val_dataset = DroneDataset(
        root, args.split, get_val_transforms(),
        skip_difficult=not args.include_difficult,
        # Evaluate against the same box population training used, or mAP compares
        # predictions to a different ground truth than the model was fitted to.
        min_score=None if args.include_difficult else cfg.min_score,
        max_area_frac=cfg.max_area_frac,
        binary=args.binary, label_map=active_map,
    )
    if args.limit:
        val_dataset.samples = val_dataset.samples[:args.limit]

    described = val_dataset.describe()
    emit(f"{args.split.title()} set: {described['total']} images "
         f"({described['positive']} with drones, "
         f"{described['negative']} negatives)")

    class_ids = sorted(active_map.values())

    # -------------------------------------------------------- clear baseline
    print(f"\nRunning inference on {args.split} set...")
    cache_path = Path(args.cache) if args.cache else None
    predictions, ground_truths = [], []
    start_index = 0
    elapsed_prior = 0.0
    cache_identity = {
        "root": str(root.resolve()), "split": args.split,
        "weights": str(Path(weights).resolve()),
        "binary": args.binary, "images": len(val_dataset),
    }
    if cache_path and cache_path.exists():
        saved = torch.load(cache_path, map_location="cpu", weights_only=False)
        if saved.get("identity") != cache_identity:
            raise SystemExit("prediction cache belongs to a different run: "
                             + str(cache_path))
        predictions = saved.get("predictions", [])
        ground_truths = saved.get("ground_truths", [])
        start_index = min(int(saved.get("next_index", 0)), len(val_dataset))
        elapsed_prior = float(saved.get("inference_seconds", 0.0))
        emit(f"Resuming prediction cache at {start_index}/{len(val_dataset)}")

    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_started = time.perf_counter()

    def save_prediction_cache(done, cached_predictions, cached_ground_truths):
        if cache_path is None:
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
        elapsed = elapsed_prior + time.perf_counter() - inference_started
        torch.save({
            "identity": cache_identity, "next_index": done,
            "inference_seconds": elapsed,
            "predictions": cached_predictions,
            "ground_truths": cached_ground_truths,
        }, temporary)
        os.replace(temporary, cache_path)

    predictions, ground_truths = collect_predictions(
        model, val_dataset, device,
        min(args.score_thresh, args.fpr_thresh, 0.10),
        start_index=start_index, all_predictions=predictions,
        all_ground_truths=ground_truths,
        checkpoint_callback=save_prediction_cache if cache_path else None,
        checkpoint_every=200)
    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_seconds = (elapsed_prior + time.perf_counter()
                         - inference_started)

    mAP_50, aps_50 = compute_mAP(predictions, ground_truths, class_ids,
                                 iou_threshold=0.5)
    mAP_50_95 = compute_mAP_range(predictions, ground_truths, class_ids)

    emit("\n=== Evaluation Results ===")
    emit(f"mAP@0.5      : {mAP_50:.4f}")
    emit(f"mAP@0.5:0.95 : {mAP_50_95:.4f}")
    if len(val_dataset):
        emit(f"End-to-end inference: {inference_seconds:.2f}s "
             f"({1000 * inference_seconds / len(val_dataset):.1f} ms/image, "
             f"{len(val_dataset) / inference_seconds:.2f} images/s)")
    emit("\nPer-class AP@0.5:")
    width = max((len(n) for n in active_map), default=8)
    for cls_id in class_ids:
        ap = aps_50[cls_id]
        name = active_names[cls_id]
        if ap is None:
            emit(f"  {name:<{width}}: n/a (no ground truth in this split)")
        else:
            emit(f"  {name:<{width}}: {ap:.4f}")

    # ------------------------------------------------------ false positives
    neg_subset = val_dataset.negatives_only()
    if len(neg_subset):
        # Reuse the aligned baseline predictions. The old implementation ran
        # every negative through the model again, then repeated that inference
        # once per category; it produced the same statistic at 2-3x the cost.
        negative_indices = [i for i, sample in enumerate(val_dataset.samples)
                            if sample["is_negative"]]

        def hits_for(indices, threshold=args.fpr_thresh):
            if not indices:
                return 0
            return sum(bool((predictions[i]["scores"] >= threshold).any())
                       for i in indices)

        def rate_for(indices, threshold=args.fpr_thresh):
            return hits_for(indices, threshold) / len(indices) if indices else 0.0

        fpr = rate_for(negative_indices)
        false_positive_count = hits_for(negative_indices)
        fpr_low, fpr_high = wilson_interval(
            false_positive_count, len(negative_indices))
        emit(f"\nFalse Positive Rate @ {args.fpr_thresh}: "
             f"{fpr:.4f} ({fpr * 100:.1f}%, "
             f"{false_positive_count}/{len(negative_indices)})")
        emit(f"  95% Wilson interval: {fpr_low * 100:.1f}% to "
             f"{fpr_high * 100:.1f}%")
        emit(f"  over {len(neg_subset)} held-out negatives")

        # Which kinds of negative actually fool it. A detector that only trips on
        # birds needs different data than one that trips on empty sky, and the
        # aggregate rate hides that entirely.
        by_category = {}
        for sample in neg_subset.samples:
            by_category.setdefault(
                sample["meta"].get("category", "?"), []).append(sample)
        if len(by_category) > 1:
            emit("\n  False positives by negative category:")
            for category, samples in sorted(by_category.items()):
                indices = [i for i in negative_indices
                           if val_dataset.samples[i]["meta"].get("category", "?")
                           == category]
                rate = rate_for(indices)
                hits = hits_for(indices)
                emit(f"    {category:<24} {rate * 100:5.1f}%  "
                     f"({hits}/{len(samples)})")

        # Image-level operating points complement detection mAP. These answer
        # the deployment question, "does this frame contain any drone?", while
        # mAP above separately measures whether the box is correctly localized.
        positive_indices = [i for i, sample in enumerate(val_dataset.samples)
                            if not sample["is_negative"]]
        emit("\nImage-level operating-point sweep:")
        emit(f"  {'threshold':>9} {'precision':>10} {'recall':>8} "
             f"{'F1':>8} {'specificity':>12} {'FP':>7} {'FN':>7}")
        for threshold in (0.10, 0.25, 0.50, 0.75, 0.90):
            tp = hits_for(positive_indices, threshold)
            fp = hits_for(negative_indices, threshold)
            fn = len(positive_indices) - tp
            tn = len(negative_indices) - fp
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = (2 * precision * recall / (precision + recall)
                  if precision + recall else 0.0)
            specificity = tn / (tn + fp) if tn + fp else 0.0
            emit(f"  {threshold:9.2f} {precision:10.4f} {recall:8.4f} "
                 f"{f1:8.4f} {specificity:12.4f} {fp:7d} {fn:7d}")

    # ---------------------------------------------------- adverse conditions
    if args.adverse:
        names = args.conditions or list(ADVERSE_SUITE)
        unknown = [n for n in names if n not in ADVERSE_SUITE]
        if unknown:
            raise SystemExit(f"unknown condition(s): {unknown}; "
                             f"known: {sorted(ADVERSE_SUITE)}")

        positives = val_dataset.positives_only()
        emit(f"\n=== Adverse-condition robustness "
             f"({len(positives)} images with drones) ===")
        emit(f"{'condition':<16} {'mAP@0.5':>9} {'mAP@.5:.95':>11} "
             f"{'retained':>9}")

        baseline = None
        for name in names:
            view = positives.with_transforms(ADVERSE_SUITE[name])
            print(f"\n  {name}...")
            preds, gts = collect_predictions(
                model, view, device, args.score_thresh)
            m50, _ = compute_mAP(preds, gts, class_ids, iou_threshold=0.5)
            m5095 = compute_mAP_range(preds, gts, class_ids)
            if baseline is None and name == "clear":
                baseline = m50
            retained = (f"{100 * m50 / baseline:7.1f}%"
                        if baseline else "      --")
            emit(f"{name:<16} {m50:9.4f} {m5095:11.4f} {retained:>9}")

        if baseline:
            emit("\n  'retained' is mAP@0.5 as a percentage of the clear-weather "
                 "baseline.")

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\nReport written to " + str(path))


if __name__ == "__main__":
    main()
