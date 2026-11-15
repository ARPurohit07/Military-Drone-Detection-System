"""Build the binary drone-detection dataset from audited external sources.

Sources:
  * UAV_SMID: 13,928 real images with VOC boxes for drone and distractors
  * banderastepan/drone-detection: 9,000 CC0 synthetic military-drone images
  * detections/: deployment incidents mined from the previous model

Only UAV_SMID objects named ``drone`` are foreground. Birds, aeroplanes,
helicopters and bombs are intentionally retained as empty-annotation hard
negatives. Synthetic imagery is train-only so benchmark metrics are measured on
real images and cannot benefit from synthetic background leakage.
"""

import argparse
import json
import os
import random
import re
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import cv2

from data.build.assemble import _write_voc
from data.build.hashing import (DuplicateIndex, dhash_file, group_duplicates,
                                ncc)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SMID_ROOT = Path("dataset_sources/UAV_SMID/dataset")
SYNTH_ROOT = Path("dataset_sources/military_synthetic/dataset")
BIRD_FLOCK_ROOT = Path("dataset_sources/bird_flocks")
DEPLOYMENT_ROOT = Path("detections")
DEFAULT_OUT = Path("drone_dataset_binary_v4_birds")


def link_or_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def image_for(directory, stem):
    for extension in IMAGE_EXTENSIONS:
        candidate = directory / f"{stem}{extension}"
        if candidate.exists():
            return candidate
    return None


def parse_voc(path):
    root = ET.parse(path).getroot()
    size = root.find("size")
    width = int(float(size.findtext("width", "0")))
    height = int(float(size.findtext("height", "0")))
    objects = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip().lower()
        box = obj.find("bndbox")
        if box is None:
            continue
        coords = [float(box.findtext(k, "0"))
                  for k in ("xmin", "ymin", "xmax", "ymax")]
        if coords[2] <= coords[0] or coords[3] <= coords[1]:
            continue
        objects.append((name, coords))
    return width, height, objects


def add_sample(out, split, source, stem, width, height, boxes, meta, index):
    suffix = source.suffix.lower()
    filename = f"{stem}{suffix}"
    image_dest = out / "images" / split / filename
    annotation_dest = out / "annotations" / split / f"{stem}.xml"
    link_or_copy(source, image_dest)
    # Extremely large boxes are valid close-range examples, but the production
    # loader intentionally excludes them (its max_box_area_fraction is 0.75).
    # Mark them difficult in VOC as well so validation and training agree.
    frame_area = max(1.0, float(width * height))
    voc_objects = []
    for box in boxes:
        box_area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
        voc_objects.append((box, 1.0, int(box_area / frame_area > 0.75)))
    kind = "drone" if boxes else "negative"
    _write_voc(annotation_dest, filename, width, height, "drone",
               voc_objects, dict(meta, split=split, kind=kind))
    index.append({
        "filename": filename, "split": split,
        "kind": kind,
        "class_name": "drone" if boxes else None,
        **meta,
    })


def import_smid(out, index, stats, seed=1337):
    records = []
    for original_split in ("train", "val", "test"):
        source_split = SMID_ROOT / original_split
        for xml in sorted((source_split / "annotations").glob("*.xml")):
            image = image_for(source_split / "images", xml.stem)
            if image is None:
                stats["smid_missing_image"] += 1
                continue
            width, height, objects = parse_voc(xml)
            classes = sorted({name for name, _ in objects})
            boxes = [box for name, box in objects if name == "drone"]
            category = "smid_drone" if boxes else (
                "smid_" + "_".join(classes) if classes else "smid_empty")
            records.append({
                "image": image, "xml": xml, "width": width, "height": height,
                "boxes": boxes, "classes": classes, "category": category,
                "original_split": original_split, "dhash": dhash_file(image),
            })

    # The publisher's nominal split contains hundreds of confirmed visual
    # duplicates across train/val/test. Cluster globally and keep one best copy
    # before drawing a fresh stratified split, otherwise benchmark mAP is
    # inflated by images the model effectively saw during training.
    gray_cache = {}

    def gray(i):
        if i not in gray_cache:
            image = cv2.imread(str(records[i]["image"]), cv2.IMREAD_GRAYSCALE)
            gray_cache[i] = (None if image is None else
                             cv2.resize(image, (64, 64),
                                        interpolation=cv2.INTER_AREA))
        return gray_cache[i]

    groups = group_duplicates(
        [record["dhash"] for record in records],
        verify=lambda i, j: ncc(gray(i), gray(j)) >= 0.92,
        chunk_rows=64,
    )
    unique = []
    for group in groups:
        best = max(group, key=lambda i: (
            records[i]["width"] * records[i]["height"],
            records[i]["image"].stat().st_size,
        ))
        unique.append(records[best])
    stats["smid_duplicate_removed"] = len(records) - len(unique)

    by_category = defaultdict(list)
    for record in unique:
        by_category[record["category"]].append(record)
    rng = random.Random(seed)
    assignments = {}
    for category in sorted(by_category):
        members = sorted(by_category[category], key=lambda r: str(r["image"]))
        rng.shuffle(members)
        n = len(members)
        n_test = max(1, round(0.1 * n)) if n >= 3 else 0
        n_val = max(1, round(0.1 * n)) if n >= 2 else 0
        while n_test + n_val >= n:
            if n_test:
                n_test -= 1
            elif n_val:
                n_val -= 1
        for rank, record in enumerate(members):
            assignments[id(record)] = (
                "test" if rank < n_test else
                "val" if rank < n_test + n_val else "train"
            )

    for record in unique:
        split = assignments[id(record)]
        image = record["image"]
        classes = record["classes"]
        boxes = record["boxes"]
        add_sample(
            out, split, image,
            f"smid_{record['original_split']}_{record['xml'].stem}",
            record["width"], record["height"],
            boxes,
            {"provenance": "external_annotated",
             "category": record["category"],
             "source_dataset": "UAV_SMID",
             "source_classes": ",".join(classes), "license": "CC BY 4.0"},
            index,
        )
        stats[f"smid_{split}"] += 1
        stats["smid_positive" if boxes else "smid_negative"] += 1
        stats["drone_boxes"] += len(boxes)


def import_synthetic(out, index, stats):
    images = SYNTH_ROOT / "images"
    labels = SYNTH_ROOT / "labels"
    for image in sorted(p for p in images.iterdir()
                        if p.suffix.lower() in IMAGE_EXTENSIONS):
        label = labels / f"{image.stem}.txt"
        if not label.exists():
            stats["synthetic_missing_label"] += 1
            continue
        array = cv2.imread(str(image))
        if array is None:
            stats["synthetic_unreadable"] += 1
            continue
        height, width = array.shape[:2]
        boxes = []
        for line in label.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5 or parts[0] != "0":
                continue
            cx, cy, bw, bh = map(float, parts[1:])
            x1 = max(0.0, (cx - bw / 2) * width)
            y1 = max(0.0, (cy - bh / 2) * height)
            x2 = min(float(width), (cx + bw / 2) * width)
            y2 = min(float(height), (cy + bh / 2) * height)
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
        if not boxes:
            stats["synthetic_empty_label"] += 1
            continue
        add_sample(
            out, "train", image, f"synthetic_{image.stem}", width, height,
            boxes,
            {"provenance": "external_synthetic", "category": "military_drone",
             "source_dataset": "banderastepan/drone-detection",
             "source_classes": "drone", "license": "CC0"},
            index,
        )
        stats["synthetic_train"] += 1
        stats["drone_boxes"] += len(boxes)


def deployment_group(path):
    """Group adjacent video frames by camera and minute to prevent leakage."""
    match = re.search(r"(cam\d+_)?frame_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2})", path.stem)
    return match.group(0) if match else path.stem


def import_deployment_negatives(out, index, stats, seed=1337):
    candidates = sorted(p for p in DEPLOYMENT_ROOT.glob("*")
                        if p.suffix.lower() in IMAGE_EXTENSIONS)
    duplicate_index = DuplicateIndex()
    unique = []
    for image in candidates:
        hsh = dhash_file(image)
        if hsh is None or duplicate_index.find(hsh) is not None:
            stats["deployment_duplicate"] += 1
            continue
        duplicate_index.add(hsh, image.name)
        unique.append(image)

    groups = defaultdict(list)
    for image in unique:
        groups[deployment_group(image)].append(image)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n_test = max(1, round(0.2 * len(keys))) if keys else 0
    test_keys = set(keys[:n_test])

    for key in keys:
        split = "test" if key in test_keys else "train"
        for image in groups[key]:
            array = cv2.imread(str(image))
            if array is None:
                stats["deployment_unreadable"] += 1
                continue
            height, width = array.shape[:2]
            stem = "deployment_" + re.sub(r"[^a-zA-Z0-9_-]", "_", image.stem)
            add_sample(
                out, split, image, stem, width, height, [],
                {"provenance": "deployment_mined",
                 "category": "deployment_false_positive",
                 "source_dataset": "camera_incidents", "source_classes": "",
                 "license": "project-owned"},
                index,
            )
            stats[f"deployment_{split}"] += 1


def import_bird_flock_negatives(out, index, stats, seed=1337):
    """Import licensed external flock silhouettes with a held-out subset."""
    manifest = BIRD_FLOCK_ROOT / "manifest.jsonl"
    if not manifest.exists():
        return
    rows = [json.loads(line) for line in manifest.open(encoding="utf-8")
            if line.strip()]
    random.Random(seed).shuffle(rows)
    test_count = max(1, round(0.2 * len(rows))) if rows else 0
    for position, row in enumerate(rows):
        source = Path(row["path"])
        if not source.exists():
            stats["bird_flock_missing"] += 1
            continue
        array = cv2.imread(str(source))
        if array is None:
            stats["bird_flock_unreadable"] += 1
            continue
        height, width = array.shape[:2]
        split = "test" if position < test_count else "train"
        stem = "commons_" + re.sub(r"[^a-zA-Z0-9_-]", "_", source.stem)
        add_sample(
            out, split, source, stem, width, height, [],
            {"provenance": "external_hard_negative",
             "category": "bird_flock_commons",
             "source_dataset": "Wikimedia Commons",
             "source_classes": "bird",
             "source_page": row.get("source_page"),
             "license": row.get("license"),
             "license_url": row.get("license_url")},
            index,
        )
        stats[f"bird_flock_{split}"] += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"output already exists and is not empty: {out}")

    for split in ("train", "val", "test"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "annotations" / split).mkdir(parents=True, exist_ok=True)

    index, stats = [], Counter()
    import_smid(out, index, stats)
    import_synthetic(out, index, stats)
    import_deployment_negatives(out, index, stats)
    import_bird_flock_negatives(out, index, stats)

    (out / "classes.json").write_text(json.dumps({
        "scheme": "drone (binary)", "label_map": {"drone": 1},
        "sources": {
            "UAV_SMID": {"doi": "10.17632/3k3hjc7rkt.1",
                         "license": "CC BY 4.0"},
            "banderastepan/drone-detection": {"license": "CC0"},
            "camera_incidents": {"license": "project-owned"},
            "Wikimedia_bird_flocks": {"licenses": "per-image manifest"},
        },
    }, indent=2), encoding="utf-8")
    with (out / "index.jsonl").open("w", encoding="utf-8") as handle:
        for row in index:
            handle.write(json.dumps(row) + "\n")

    split_counts = Counter(row["split"] for row in index)
    kind_counts = Counter(row["kind"] for row in index)
    report = [
        "Binary drone dataset v3", "=======================", "",
        f"total images: {len(index)}",
        *(f"{split}: {split_counts[split]}" for split in ("train", "val", "test")),
        f"positive images: {kind_counts['drone']}",
        f"negative images: {kind_counts['negative']}",
        f"drone boxes: {stats['drone_boxes']}", "", "source counters:",
        *(f"  {key}: {value}" for key, value in sorted(stats.items())),
    ]
    (out / "BUILD_REPORT.txt").write_text("\n".join(report) + "\n",
                                           encoding="utf-8")
    print("\n".join(report))
    if len(index) < 15_000:
        raise SystemExit("dataset did not reach the required 15,000 images")


if __name__ == "__main__":
    main()
