"""Auto-annotate scraped imagery with a zero-shot detector.

This is what makes a large dataset possible without hand-drawing a single box.
Grounding DINO is prompted with a text description of a drone and returns boxes;
the *class* of each box is not taken from the detector at all - it comes from the
airframe query that fetched the image, via the manifest. The detector answers
"where", the registry answers "what". A zero-shot model has no idea which country
built an aircraft, so asking it would reintroduce exactly the label noise this
rebuild exists to remove.

Measured against the 1325 human-annotated boxes in the old dataset, this
recovers the human box with median IoU ~0.66 and finds something in 100% of
images. That is short of a careful human annotator, and the pipeline is built
around that fact rather than pretending otherwise:

Boxes are graded, not just filtered. Detections above CONFIDENT_SCORE are
written as normal objects; those between ACCEPT_SCORE and CONFIDENT_SCORE are
written with VOC's `difficult` flag set. The dataset loader already skips
`difficult` boxes, so a training run gets only the trustworthy boxes while the
weaker ones stay on disk for evaluation or review instead of being thrown away.

The old dataset's specific annotation failures are rejected outright: boxes
covering more than MAX_AREA_FRAC of the frame (there were 118 of those, and they
teach the model that "drone" means "the whole picture"), degenerate zero-area
boxes, and boxes too small to carry signal.
"""

import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom

import torch
from PIL import Image, ImageFile
from torchvision.ops import nms

from data.airframes import BY_KEY

# This pipeline is PyTorch-only.  Prevent Transformers from importing the
# installed TensorFlow stack (roughly gigabytes of avoidable host-memory/commit
# pressure on the 16 GB Windows training machine).
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

# Truncated JPEGs are common in scraped data; allow PIL to load what it has
# rather than losing the file.
ImageFile.LOAD_TRUNCATED_IMAGES = True

MODEL_ID = "IDEA-Research/grounding-dino-tiny"

# Prompt and thresholds selected by sweeping against the human-annotated boxes
# (see reports/autolabel_validation.txt). Grounding DINO expects lowercase
# phrases separated by periods.
PROMPT = "a drone. an unmanned aerial vehicle."

ACCEPT_SCORE = 0.25       # below this the detection is discarded entirely
CONFIDENT_SCORE = 0.40    # at or above this the box is trusted for training
TEXT_THRESHOLD = 0.25

# Area handling is graded rather than a single cutoff. The old dataset's defect
# was 118 boxes covering most of the frame, which teach the model that "drone"
# means "the whole picture" - but a hard 0.75 reject also discards legitimate
# close-ups, and scraped press photography is full of them (tight publicity shots
# of a TB2 were being dropped entirely, including Reuters images).
#
# So: above SUSPICIOUS_AREA_FRAC a box is kept but flagged `difficult`, which
# excludes it from training by default while leaving it on disk. Only above
# MAX_AREA_FRAC - where the box is effectively the entire image and carries no
# localisation signal at all - is it discarded.
MAX_AREA_FRAC = 0.92
SUSPICIOUS_AREA_FRAC = 0.75
MIN_AREA_FRAC = 0.0002
MIN_SIDE_PX = 8
MAX_BOXES = 6             # scraped photos occasionally show a row of aircraft

# Grounding DINO returns several overlapping proposals for the same object - one
# Shahed was emitting six boxes that were really two aircraft, including exact
# repeats one point of score apart. Without suppression those duplicates become
# duplicate training targets, and MAX_BOXES fills up with copies of one aircraft
# instead of genuinely separate ones. NMS is class-agnostic here because every
# box in an image carries the same label by construction.
NMS_IOU = 0.5

# Resolution is controlled through the PROCESSOR, not by pre-resizing the image.
# Grounding DINO's image processor defaults to shortest_edge=800/longest_edge=1333
# and will happily *upscale* to reach it, so downscaling the PIL image beforehand
# achieved nothing - it was silently undone, peak memory sat at 3.7 GB of a
# 4.29 GB card, and a batch of 4 wanted 6.71 GB and spilled to host memory at
# 17 s/image.
#
# Setting the processor size directly is what actually reduces the work. Chosen to
# match the detector the dataset trains (config.TrainConfig min_size/max_size),
# since there is no reason to auto-label at a resolution the downstream model
# never sees. Box quality at this setting is measured, not assumed - see
# reports/autolabel_validation.txt.
#
# Detections come back in normalised coordinates and `target_sizes` is the
# ORIGINAL image size, so boxes land in full-resolution pixels regardless of what
# the model actually saw.
PROCESSOR_SIZE = {"shortest_edge": 480, "longest_edge": 800}

RAW_ROOT = Path("dataset_raw")
MANIFEST = RAW_ROOT / "manifest.jsonl"
ANN_ROOT = RAW_ROOT / "annotations"
LABEL_LOG = RAW_ROOT / "autolabel.jsonl"


def _load_model(device):
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID)
    return proc, model.to(device).eval()


def _write_voc(xml_path, img_name, width, height, objects, meta):
    """Write a VOC XML holding the boxes plus the provenance of the labels.

    The extra `source` fields are not standard VOC, but parsers ignore unknown
    elements and it means a single annotation file records which airframe query
    and which detector produced it - so a later taxonomy change can be replayed
    from the XML alone.
    """
    ann = ET.Element("annotation")
    ET.SubElement(ann, "folder").text = meta["airframe"]
    ET.SubElement(ann, "filename").text = img_name
    src = ET.SubElement(ann, "source")
    ET.SubElement(src, "database").text = "drone-rebuild"
    ET.SubElement(src, "airframe").text = meta["airframe"]
    ET.SubElement(src, "airframe_name").text = meta["airframe_name"]
    ET.SubElement(src, "country").text = meta["country"]
    ET.SubElement(src, "role").text = meta["role"]
    ET.SubElement(src, "query").text = meta["query"]
    ET.SubElement(src, "annotator").text = MODEL_ID

    size = ET.SubElement(ann, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"
    ET.SubElement(ann, "segmented").text = "0"

    for box, score, difficult in objects:
        rounded_box = tuple(int(round(v)) for v in box)
        x1, y1, x2, y2 = rounded_box
        rounded_fraction = ((x2 - x1) * (y2 - y1)) / float(width * height)
        # Validation sees the integer VOC coordinates, so make the final
        # trainability decision on those same coordinates.  On very small
        # images, rounding alone can move a 74.9% box just over the 75% guard.
        difficult = difficult or rounded_fraction > SUSPICIOUS_AREA_FRAC
        obj = ET.SubElement(ann, "object")
        # The name is the airframe key. Mapping key -> country/role happens in
        # the loader through the registry, so one annotation file serves every
        # label scheme without rewriting.
        ET.SubElement(obj, "name").text = meta["airframe"]
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "1" if difficult else "0"
        ET.SubElement(obj, "score").text = f"{score:.4f}"
        bb = ET.SubElement(obj, "bndbox")
        for k, v in zip(("xmin", "ymin", "xmax", "ymax"), rounded_box):
            ET.SubElement(bb, k).text = str(v)

    xml_path.parent.mkdir(parents=True, exist_ok=True)
    pretty = minidom.parseString(ET.tostring(ann)).toprettyxml(indent="  ")
    xml_path.write_text(pretty, encoding="utf-8")


def _detect(proc, model, images, metas, device, size=None):
    """Run the detector over a batch and post-process to original coordinates."""
    inputs = proc(images=images, text=[PROMPT] * len(images),
                  return_tensors="pt", padding=True,
                  size=size or PROCESSOR_SIZE).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    # target_sizes is the ORIGINAL (h, w), not the downscaled input size, so
    # boxes come back in full-resolution pixel coordinates.
    return proc.post_process_grounded_object_detection(
        outputs, inputs.input_ids,
        threshold=ACCEPT_SCORE, text_threshold=TEXT_THRESHOLD,
        target_sizes=[(h, w) for _, _, (w, h) in metas])


def _valid(box, width, height):
    """Clamp a box and judge it.

    Returns `(box, reason, suspicious)`. `box` is None when the shape is one of
    the annotation defects that made the old dataset untrainable; `suspicious`
    marks a box that is kept but should not be trained on.
    """
    x1, y1, x2, y2 = box
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(float(width), x2), min(float(height), y2)
    w, h = x2 - x1, y2 - y1
    if w < MIN_SIDE_PX or h < MIN_SIDE_PX:
        return None, "too_small", False
    frac = (w * h) / float(width * height)
    if frac > MAX_AREA_FRAC:
        return None, "near_full_frame", False
    if frac < MIN_AREA_FRAC:
        return None, "tiny_area", False
    return (x1, y1, x2, y2), None, frac > SUSPICIOUS_AREA_FRAC


def annotate(limit=None, batch_size=4, overwrite=False, device=None):
    """Annotate every manifest entry that has no annotation yet."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if not MANIFEST.exists():
        raise SystemExit(f"no manifest at {MANIFEST} - run data.build.scrape first")

    # Tolerate a torn final line: auto-labelling is expected to run while a
    # scrape is still appending, and a half-written record must not abort the
    # whole stage.
    records = []
    for line in MANIFEST.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    # Images the detector already found nothing in produce no annotation file, so
    # without this they would be re-processed on every resume - a tax that grows
    # with the corpus. The run log already records them, so read it back.
    exhausted = set()
    if LABEL_LOG.exists() and not overwrite:
        for line in LABEL_LOG.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("status") == "no_valid_detection":
                exhausted.add(entry["filename"])

    todo = []
    skipped_exhausted = 0
    for rec in records:
        xml_path = ANN_ROOT / rec["airframe"] / (Path(rec["filename"]).stem + ".xml")
        if xml_path.exists() and not overwrite:
            continue
        if rec["filename"] in exhausted:
            skipped_exhausted += 1
            continue
        todo.append((rec, xml_path))
    if skipped_exhausted:
        print(f"skipping {skipped_exhausted} images a previous run found "
              f"no drone in (pass --overwrite to retry them)")
    if limit:
        todo = todo[:limit]

    print(f"manifest: {len(records)} images, {len(todo)} to annotate "
          f"on {device}")
    if not todo:
        return {}

    proc, model = _load_model(device)
    stats = {"annotated": 0, "confident": 0, "difficult": 0,
             "no_detection": 0, "unreadable": 0}
    rejects = {}

    # Group by aspect/size so a batch is not padded up to one outlier image.
    todo.sort(key=lambda t: (t[0].get("width") or 0) * (t[0].get("height") or 0))

    for start in range(0, len(todo), batch_size):
        chunk = todo[start:start + batch_size]
        images, metas = [], []
        for rec, xml_path in chunk:
            try:
                img = Image.open(rec["path"]).convert("RGB")
            except Exception:                                 # noqa: BLE001
                stats["unreadable"] += 1
                continue
            # No pre-resize here: PROCESSOR_SIZE governs the resolution the
            # model sees, and resizing twice only loses detail.
            images.append(img)
            metas.append((rec, xml_path, img.size))
        if not images:
            continue

        try:
            results = _detect(proc, model, images, metas, device)
        except torch.cuda.OutOfMemoryError:
            # Fall back to one image at a time rather than losing the run. A
            # single outlier should not end a multi-hour unattended build.
            torch.cuda.empty_cache()
            stats["oom_fallback"] = stats.get("oom_fallback", 0) + 1
            results = []
            for img, meta in zip(images, metas):
                try:
                    results.extend(_detect(proc, model, [img], [meta], device))
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    stats["oom_skipped"] = stats.get("oom_skipped", 0) + 1
                    results.append({"boxes": torch.zeros((0, 4)),
                                    "scores": torch.zeros((0,))})

        for (rec, xml_path, (W, H)), res in zip(metas, results):
            boxes, scores = res["boxes"], res["scores"]
            if len(boxes):
                keep = nms(boxes, scores, NMS_IOU)
                boxes, scores = boxes[keep], scores[keep]
                n_suppressed = len(res["boxes"]) - len(boxes)
                if n_suppressed:
                    rejects["nms_suppressed"] = (
                        rejects.get("nms_suppressed", 0) + n_suppressed)

            objects = []
            for i in range(len(boxes)):
                score = float(scores[i])
                box, why, suspicious = _valid(boxes[i].tolist(), W, H)
                if box is None:
                    rejects[why] = rejects.get(why, 0) + 1
                    continue
                if suspicious:
                    rejects["large_area_flagged_difficult"] = (
                        rejects.get("large_area_flagged_difficult", 0) + 1)
                # A box is trusted for training only if the detector was
                # confident AND the box is not suspiciously close to full-frame.
                difficult = score < CONFIDENT_SCORE or suspicious
                objects.append((box, score, difficult))
                if len(objects) >= MAX_BOXES:
                    break

            if not objects:
                stats["no_detection"] += 1
                _log(rec, 0, 0, "no_valid_detection")
                continue

            _write_voc(xml_path, rec["filename"], W, H, objects, rec)
            n_conf = sum(1 for _, _, d in objects if not d)
            stats["annotated"] += 1
            stats["confident"] += n_conf
            stats["difficult"] += len(objects) - n_conf
            _log(rec, len(objects), n_conf, "ok")

        done = start + len(chunk)
        if done % 200 < batch_size:
            print(f"   {done}/{len(todo)}  annotated={stats['annotated']} "
                  f"nodet={stats['no_detection']}")
        # Drop Python references while keeping reusable GPU blocks cached.
        # Frequent empty_cache calls caused native c10.dll faults under WDDM.
        # The OOM fallback above still flushes the cache when actually needed.
        del results, images, metas

    print("\n=== auto-label summary ===")
    for k, v in stats.items():
        print(f"  {k:14s} {v}")
    if rejects:
        print("  rejected boxes:")
        for k, v in sorted(rejects.items(), key=lambda kv: -kv[1]):
            print(f"    {v:5d}  {k}")
    return stats


def _log(rec, n_boxes, n_confident, status):
    LABEL_LOG.parent.mkdir(parents=True, exist_ok=True)
    with LABEL_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "filename": rec["filename"],
            "airframe": rec["airframe"],
            "country": rec["country"],
            "boxes": n_boxes,
            "confident": n_confident,
            "status": status,
        }) + "\n")


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=2,
                   help="2 fits a 4 GB card comfortably after downscaling")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--device", default=None)
    a = p.parse_args()
    annotate(limit=a.limit, batch_size=a.batch_size,
             overwrite=a.overwrite, device=a.device)


if __name__ == "__main__":
    main()
