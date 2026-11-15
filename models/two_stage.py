"""Two-stage Faster R-CNN pipeline for detection and origin attribution.

Stage 1 is a one-class detector answering "is there a drone, and where?" Stage
2 is an 18-class detector that proposes the airframe's design/manufacturing
country. Its boxes are paired to stage-1 boxes by IoU and returned as ranked
candidates. A legacy crop classifier remains supported only as a fallback.

The split reflects measured reliability: the held-out binary detector reached
0.9503 mAP@0.5 with 7.82% image-level FPR, while country attribution reached
33.6% top-1 and 59.1% top-3
on localized objects. Either model can therefore be retrained, calibrated, or
replaced independently, and uncertain country output is never required for a
binary alert.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import box_iou, nms

from config import TrainConfig
from models.faster_rcnn import build_model
from models.inference import resolve_device

# Must match tools/crop_classifier.py, or the classifier sees crops framed
# differently from the ones it was trained on.
CROP_SIZE = 224
BOX_PAD = 0.15
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _finite_metric(value):
    """Return a JSON-safe checkpoint metric (old runs sometimes stored NaN)."""
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


class TwoStageDroneDetector:
    """Detector + optional country classifier.

    The classifier is optional on purpose: with no stage-2 weights the pipeline
    still detects drones, which is the part that works well.
    """

    def __init__(self, detector_path, classifier_path=None, device=None,
                 score_thresh=0.90, nms_thresh=0.4, country_min_share=0.0,
                 country_detector_path=None, country_iou_thresh=0.3):
        self.device = resolve_device(device)
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.country_min_share = country_min_share
        self.country_iou_thresh = country_iou_thresh

        cfg = TrainConfig()
        self.detector = build_model(
            num_classes=1, pretrained=False,
            min_size=cfg.min_size, max_size=cfg.max_size)
        blob = torch.load(detector_path, map_location=self.device,
                          weights_only=False)
        state = blob.get("model_state_dict", blob) if isinstance(blob, dict) else blob
        self.detector.load_state_dict(state)
        self.detector.to(self.device).eval()
        self.detector_meta = {
            k: blob.get(k) for k in ("epoch", "val_loss", "label_scheme")
        } if isinstance(blob, dict) else {}
        if self.detector_meta:
            self.detector_meta["val_loss"] = _finite_metric(
                self.detector_meta.get("val_loss")
            )

        self.classifier = None
        self.country_detector = None
        self.classes = None
        if country_detector_path and Path(country_detector_path).exists():
            self._load_country_detector(country_detector_path)
        elif classifier_path and Path(classifier_path).exists():
            self._load_classifier(classifier_path)

    def _load_country_detector(self, path):
        blob = torch.load(path, map_location=self.device, weights_only=False)
        label_map = blob.get("class_map")
        if not label_map:
            raise ValueError(f"country checkpoint has no class_map: {path}")
        cfg = TrainConfig()
        self.country_names = {int(value): name
                              for name, value in label_map.items()}
        model_cfg = blob.get("model_config", {})
        self.country_detector = build_model(
            num_classes=len(label_map), pretrained=False,
            min_size=int(model_cfg.get("min_size", cfg.min_size)),
            max_size=int(model_cfg.get("max_size", cfg.max_size)))
        self.country_detector.load_state_dict(blob["model_state_dict"])
        self.country_detector.to(self.device).eval()
        self.country_detector_meta = {
            "scheme": blob.get("label_scheme", "country_of_origin"),
            "epoch": blob.get("epoch"),
            "val_loss": _finite_metric(blob.get("val_loss")),
            "classes": len(label_map),
        }

    def _load_classifier(self, path):
        import timm

        blob = torch.load(path, map_location=self.device, weights_only=False)
        label_map = blob["label_map"]
        self.classes = [c for c, _ in sorted(label_map.items(),
                                             key=lambda kv: kv[1])]
        self.classifier = timm.create_model(
            blob.get("backbone", "convnext_tiny"), pretrained=False,
            num_classes=len(label_map))
        self.classifier.load_state_dict(blob["model_state_dict"])
        self.classifier.to(self.device).eval()
        self.classifier_meta = {"scheme": blob.get("scheme"),
                                "val_top1": blob.get("val_top1")}

    # ------------------------------------------------------------------ crops

    @staticmethod
    def _crop(image_bgr, box, pad=BOX_PAD):
        h, w = image_bgr.shape[:2]
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        x1 = int(max(0, x1 - bw * pad))
        y1 = int(max(0, y1 - bh * pad))
        x2 = int(min(w, x2 + bw * pad))
        y2 = int(min(h, y2 + bh * pad))
        if x2 - x1 < 2 or y2 - y1 < 2:
            return None
        crop = image_bgr[y1:y2, x1:x2]
        crop = cv2.resize(crop, (CROP_SIZE, CROP_SIZE),
                          interpolation=cv2.INTER_LINEAR)
        arr = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        return np.ascontiguousarray(arr.transpose(2, 0, 1), dtype=np.float32)

    # -------------------------------------------------------------- inference

    @torch.no_grad()
    def predict(self, image_bgr, top_k=3):
        """Detect drones and attribute them. Returns a list of detections."""
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(
            rgb.astype(np.float32).transpose(2, 0, 1) / 255.0).to(self.device)

        use_amp = self.device.type == "cuda"
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = self.detector([tensor])[0]
        boxes, scores = out["boxes"], out["scores"]
        keep = scores >= self.score_thresh
        boxes, scores = boxes[keep], scores[keep]
        if len(boxes):
            k = nms(boxes, scores, self.nms_thresh)
            boxes, scores = boxes[k], scores[k]

        results = []
        if not len(boxes):
            return results

        crops, index = [], []
        if self.country_detector is None:
            for i, box in enumerate(boxes.cpu().numpy()):
                arr = self._crop(image_bgr, box)
                if arr is not None:
                    crops.append(arr)
                    index.append(i)

        # One batched forward for every crop in the image rather than one call
        # per detection.
        probs = None
        if self.classifier is not None and crops:
            batch = torch.from_numpy(np.stack(crops)).to(self.device)
            logits = self.classifier(batch)
            probs = F.softmax(logits.float(), dim=1).cpu().numpy()

        country_out = None
        if self.country_detector is not None:
            with torch.amp.autocast("cuda", enabled=use_amp):
                country_out = self.country_detector([tensor])[0]

        for i, (box, score) in enumerate(zip(boxes.cpu().numpy(),
                                             scores.cpu().numpy())):
            det = {
                "bbox": [float(v) for v in box],
                "score": float(score),
                "label": "Drone",
                "candidates": [],
            }
            if probs is not None and i in index:
                row = probs[index.index(i)]
                order = np.argsort(-row)[:top_k]
                det["candidates"] = [
                    {"label": self.classes[j], "score": float(row[j]),
                     "share": float(row[j])}
                    for j in order
                    if float(row[j]) >= self.country_min_share
                ]
                if det["candidates"]:
                    det["country"] = det["candidates"][0]["label"]
                    det["country_confidence"] = det["candidates"][0]["share"]
            elif country_out is not None and len(country_out["boxes"]):
                reference = torch.as_tensor(box, device=self.device).unsqueeze(0)
                ious = box_iou(reference, country_out["boxes"])[0]
                nearby = (ious >= self.country_iou_thresh).nonzero().flatten()
                best_by_label = {}
                for j in nearby:
                    label_id = int(country_out["labels"][j])
                    score_value = float(country_out["scores"][j])
                    best_by_label[label_id] = max(
                        best_by_label.get(label_id, 0.0), score_value)
                ranked = sorted(best_by_label.items(), key=lambda item: -item[1])[:top_k]
                total = sum(score for _, score in ranked) or 1.0
                det["candidates"] = [
                    {"label": self.country_names.get(label, f"class_{label}"),
                     "score": score, "share": score / total}
                    for label, score in ranked
                    if score / total >= self.country_min_share
                ]
                if det["candidates"]:
                    det["country"] = det["candidates"][0]["label"]
                    det["country_confidence"] = det["candidates"][0]["share"]
            results.append(det)
        del out, tensor
        if country_out is not None:
            del country_out
        return results

    def describe(self):
        d = {"detector": self.detector_meta, "stage2": None}
        if self.country_detector is not None:
            d["stage2"] = dict(self.country_detector_meta,
                               architecture="Faster R-CNN")
        elif self.classifier is not None:
            d["stage2"] = dict(self.classifier_meta,
                               classes=len(self.classes or []))
        return d
