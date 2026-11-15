"""Detection metrics.

AP is computed by pooling every detection of a class across the whole dataset
and ranking them by score, which is what VOC/COCO mAP means. The previous
implementation computed an AP per image and averaged those, which is a
different (and much more optimistic) quantity: a single easy object in an image
scores AP 1.0 there and outweighs a crowded image the model got wrong.
"""

import torch
from tqdm import tqdm


def collect_predictions(model, dataset, device, score_threshold=0.05,
                        start_index=0, all_predictions=None,
                        all_ground_truths=None, checkpoint_callback=None,
                        checkpoint_every=200):
    """Run the model over a dataset and return aligned prediction/GT lists."""
    model.eval()
    all_predictions = [] if all_predictions is None else all_predictions
    all_ground_truths = ([] if all_ground_truths is None
                         else all_ground_truths)

    use_amp = torch.device(device).type == "cuda"
    # Evaluation previously ran Faster R-CNN entirely in FP32 even though
    # training uses AMP. On a 4 GB RTX 3050, a wide 800x1333 frame can exceed
    # VRAM and terminate the process without a useful Python traceback. AMP is
    # safe for inference and keeps evaluation aligned with deployment.
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=use_amp):
        for idx in tqdm(range(start_index, len(dataset)), desc="Evaluating",
                        total=len(dataset), initial=start_index):
            img, target = dataset[idx]
            prediction = model([img.to(device)])[0]

            keep = prediction["scores"] > score_threshold
            all_predictions.append({
                "boxes": prediction["boxes"][keep].cpu(),
                "labels": prediction["labels"][keep].cpu(),
                "scores": prediction["scores"][keep].cpu(),
            })
            all_ground_truths.append({
                "boxes": target["boxes"].cpu(),
                "labels": target["labels"].cpu(),
            })
            # Release per-image outputs promptly. Keep reusable blocks in the
            # caching allocator: repeatedly calling empty_cache caused native
            # c10.dll access violations during the long training benchmark on
            # this Windows/WDDM machine.
            del prediction, keep, img, target
            done = idx + 1
            if (checkpoint_callback is not None and checkpoint_every
                    and done % checkpoint_every == 0):
                checkpoint_callback(done, all_predictions, all_ground_truths)

    if checkpoint_callback is not None:
        checkpoint_callback(len(dataset), all_predictions, all_ground_truths)

    return all_predictions, all_ground_truths


# Backwards-compatible alias for the old name.
compute_map = collect_predictions


def box_iou(box, boxes):
    """IoU of one (4,) box against an (N, 4) tensor of boxes."""
    if len(boxes) == 0:
        return torch.zeros((0,), dtype=torch.float32)

    x1 = torch.maximum(box[0], boxes[:, 0])
    y1 = torch.maximum(box[1], boxes[:, 1])
    x2 = torch.minimum(box[2], boxes[:, 2])
    y2 = torch.minimum(box[3], boxes[:, 3])

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    return inter / (area + areas - inter + 1e-6)


def compute_ap(predictions_list, ground_truths_list, class_id, iou_threshold=0.5):
    """All-point-interpolated AP for one class, pooled across the dataset.

    Returns ``None`` when the class has no ground truth in this split, so that
    an absent class is excluded from the mean rather than dragging it to zero.
    """
    gt_boxes_by_image = []
    matched_by_image = []
    n_gt = 0
    for gt in ground_truths_list:
        mask = gt["labels"] == class_id
        boxes = gt["boxes"][mask]
        gt_boxes_by_image.append(boxes)
        matched_by_image.append(torch.zeros(len(boxes), dtype=torch.bool))
        n_gt += len(boxes)

    if n_gt == 0:
        return None

    # Pool every detection of this class across all images, then rank globally.
    records = []
    for image_idx, pred in enumerate(predictions_list):
        mask = pred["labels"] == class_id
        for box, score in zip(pred["boxes"][mask], pred["scores"][mask]):
            records.append((float(score), image_idx, box))

    if not records:
        return 0.0

    records.sort(key=lambda r: r[0], reverse=True)

    tp = torch.zeros(len(records))
    fp = torch.zeros(len(records))

    for i, (_score, image_idx, box) in enumerate(records):
        gt_boxes = gt_boxes_by_image[image_idx]
        matched = matched_by_image[image_idx]

        ious = box_iou(box, gt_boxes)
        ious[matched] = -1.0  # already claimed by a higher-scoring detection

        if len(ious) > 0 and ious.max() >= iou_threshold:
            tp[i] = 1.0
            matched[int(ious.argmax())] = True
        else:
            fp[i] = 1.0

    tp_cum = torch.cumsum(tp, dim=0)
    fp_cum = torch.cumsum(fp, dim=0)
    precision = tp_cum / (tp_cum + fp_cum)
    recall = tp_cum / n_gt

    precision = torch.cat([torch.tensor([1.0]), precision])
    recall = torch.cat([torch.tensor([0.0]), recall])

    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    ap = 0.0
    for i in range(1, len(recall)):
        if recall[i] != recall[i - 1]:
            ap += float((recall[i] - recall[i - 1]) * precision[i])

    return ap


def compute_mAP(predictions_list, ground_truths_list, class_ids, iou_threshold=0.5):
    """mAP over the given class IDs.

    ``class_ids`` is an explicit iterable of label values, not a count — the
    label space is not guaranteed to be ``range(num_classes)``.

    Returns ``(mAP, {class_id: ap_or_None})``.
    """
    aps = {
        cls_id: compute_ap(predictions_list, ground_truths_list, cls_id, iou_threshold)
        for cls_id in class_ids
    }
    present = [ap for ap in aps.values() if ap is not None]
    return (sum(present) / len(present) if present else 0.0), aps


def compute_mAP_range(predictions_list, ground_truths_list, class_ids,
                      thresholds=None):
    """COCO-style mAP averaged over IoU thresholds 0.50:0.05:0.95."""
    if thresholds is None:
        thresholds = [0.5 + 0.05 * i for i in range(10)]

    per_threshold = [
        compute_mAP(predictions_list, ground_truths_list, class_ids, t)[0]
        for t in thresholds
    ]
    return sum(per_threshold) / len(per_threshold)


def compute_false_positive_rate(model, negative_dataset, device, score_threshold=0.5):
    """Fraction of drone-free images that still produce at least one detection."""
    model.eval()
    fp_count = 0
    total = len(negative_dataset)

    use_amp = torch.device(device).type == "cuda"
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
        for idx in tqdm(range(total), desc="Computing FPR"):
            img, _ = negative_dataset[idx]
            prediction = model([img.to(device)])[0]
            if (prediction["scores"] > score_threshold).sum() > 0:
                fp_count += 1

    return fp_count / total if total > 0 else 0.0
