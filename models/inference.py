import torch
import cv2
from torchvision.ops import box_iou, nms
from torchvision.transforms import functional as F

from config import CLASS_NAMES, CLASS_MAP, LABEL_MAP_SOURCE, TrainConfig


def resolve_device(device=None):
    """Pick a torch device, defaulting to CUDA when it is available."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def load_weights(model, path, device, strict_labels=True,
                 expected_label_map=None, label_map_source=None):
    """Load weights from either checkpoint format into `model`.

    Training now saves a dict carrying the epoch, optimizer state and the label
    map it was trained against; older files are a bare `state_dict`. Both are
    accepted here so evaluation and the app work across the transition.

    When the checkpoint records its label map, it is compared against the current
    one and a mismatch raises. This check is the reason the field is saved: a
    detector loaded against a different label space does not fail, it silently
    renames every prediction - which is exactly how the old 0-based checkpoint
    could be pointed at a 1-based map and quietly mislabel everything.
    """
    blob = torch.load(path, map_location=device, weights_only=False)

    if isinstance(blob, dict) and "model_state_dict" in blob:
        saved_map = blob.get("class_map")
        expected_map = expected_label_map or CLASS_MAP
        expected_source = label_map_source or LABEL_MAP_SOURCE
        if strict_labels and saved_map and saved_map != expected_map:
            raise SystemExit(
                f"checkpoint {path} was trained on a different label space.\n"
                f"  checkpoint: {saved_map}\n"
                f"  current   : {expected_map} (from {expected_source})\n"
                f"Loading it would silently mislabel every detection."
            )
        state = blob["model_state_dict"]
        meta = {k: blob.get(k) for k in ("epoch", "val_loss", "label_scheme")}
    else:
        state = blob
        meta = {}

    model.load_state_dict(state)
    return meta


class DroneDetector:
    def __init__(self, model_path, device=None, score_thresh=0.5, nms_thresh=0.4):
        from models.faster_rcnn import build_model

        self.device = resolve_device(device)
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh

        cfg = TrainConfig()
        self.model = build_model(
            len(CLASS_MAP), pretrained=False,
            min_size=cfg.min_size, max_size=cfg.max_size,
        )
        self.checkpoint_meta = load_weights(self.model, model_path, self.device)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, image_bgr):
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        # No manual normalization: the model's GeneralizedRCNNTransform applies
        # ImageNet mean/std itself. Must stay in step with data/transforms.py.
        img_tensor = F.to_tensor(img_rgb).to(self.device)

        prediction = self.model([img_tensor])[0]

        boxes = prediction["boxes"]
        scores = prediction["scores"]
        labels = prediction["labels"]

        keep = scores >= self.score_thresh
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        # torchvision applies NMS per class inside roi_heads, so the same drone
        # can survive once per class — one airframe reported as both "Chinese"
        # and "Iranian" with overlapping boxes. Collapse those with a
        # class-agnostic pass and keep the highest-scoring label.
        if len(boxes) > 0:
            keep_idx = nms(boxes, scores, self.nms_thresh)
            boxes, scores, labels = boxes[keep_idx], scores[keep_idx], labels[keep_idx]

        # Keep the competing class hypotheses for each surviving box. The
        # country model names the origin correctly only ~1 time in 3 after
        # localization, so reporting a single label
        # presents a coin-flip as a fact. Callers get the ranked alternatives and
        # can show the uncertainty instead of hiding it.
        all_boxes = prediction["boxes"][keep]
        all_scores = prediction["scores"][keep]
        all_labels = prediction["labels"][keep]

        results = []
        for box, score, label in zip(boxes.cpu(), scores.cpu(), labels.cpu()):
            label = int(label.item())
            if len(all_boxes) > 0:
                # box_iou needs (N,4); `box` here is a single (4,) row.
                ious = box_iou(box.to(all_boxes.device).unsqueeze(0),
                               all_boxes)[0]
                near = (ious >= 0.5).nonzero().flatten()
                ranked = {}
                for i in near:
                    lab = int(all_labels[i])
                    sc = float(all_scores[i])
                    ranked[lab] = max(ranked.get(lab, 0.0), sc)
                alts = sorted(ranked.items(), key=lambda kv: -kv[1])
            else:
                alts = [(label, float(score.item()))]

            total = sum(s for _, s in alts) or 1.0
            results.append({
                "bbox": box.numpy().tolist(),
                "label": CLASS_NAMES.get(label, f"class_{label}"),
                "score": float(score.item()),
                # Normalised so they read as "given this is a drone, whose is it"
                "candidates": [
                    {"label": CLASS_NAMES.get(l, f"class_{l}"),
                     "score": s, "share": s / total}
                    for l, s in alts
                ],
            })
        return results
