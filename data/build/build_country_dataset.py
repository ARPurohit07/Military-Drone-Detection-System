"""Build a leakage-resistant country-of-origin Faster R-CNN dataset.

The country label means the airframe's design/manufacturing origin, not the
operator visible in a particular photograph. Three audited sources are merged:

* AeroScan human YOLO boxes for six UAV types (Turkey, USA and China)
* query-provenance images auto-labelled by Grounding DINO
* hard negatives from the validated binary dataset

All sources are perceptually deduplicated together before a fresh stratified
80/10/10 split. Conflicting duplicate groups are quarantined rather than voted
on. Countries below ``--min-per-country`` are reported and excluded because a
class with only a handful of pictures cannot support a defensible test metric.
"""

import argparse
import json
import os
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from xml.dom import minidom

import cv2

from data.airframes import AIRFRAMES
from data.build.hashing import dhash_file, group_duplicates, ncc


AEROSCAN_ROOT = Path("dataset_sources/aeroscan/dataset")
RAW_ROOT = Path("dataset_raw")
BINARY_ROOT = Path("drone_dataset_binary_v3_dedup2")
DEFAULT_OUT = Path("drone_dataset_country_v3")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# AeroScan class id -> (airframe key, country of design/manufacture).
AEROSCAN_UAVS = {
    4: ("bayraktar_akinci", "Turkey"),
    52: ("mq9_reaper", "USA"),
    61: ("rq4_global_hawk", "USA"),
    69: ("tb001", "China"),
    70: ("bayraktar_tb2", "Turkey"),
    81: ("wz_7", "China"),
}


def load_jsonl(path):
    rows = []
    if not path.exists():
        return rows
    for line in path.open(encoding="utf-8"):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def image_for_stem(directory, stem):
    for suffix in IMAGE_EXTENSIONS:
        path = directory / f"{stem}{suffix}"
        if path.exists():
            return path
    return None


def image_size(path):
    image = cv2.imread(str(path))
    if image is None:
        return None
    height, width = image.shape[:2]
    return width, height


def parse_voc(path):
    root = ET.parse(path).getroot()
    size = root.find("size")
    width = int(float(size.findtext("width", "0")))
    height = int(float(size.findtext("height", "0")))
    objects = []
    for obj in root.findall("object"):
        box = obj.find("bndbox")
        if box is None:
            continue
        coords = [float(box.findtext(k, "0"))
                  for k in ("xmin", "ymin", "xmax", "ymax")]
        if coords[2] <= coords[0] or coords[3] <= coords[1]:
            continue
        score = float(obj.findtext("score", "1"))
        difficult = int(obj.findtext("difficult", "0"))
        objects.append((coords, score, difficult))
    meta = {}
    source = root.find("source")
    if source is not None:
        for child in source:
            if child.text:
                meta[child.tag] = child.text
    return width, height, objects, meta


def collect_aeroscan(stats):
    records = []
    for original_split in ("train", "val", "test"):
        directory = AEROSCAN_ROOT / original_split
        for label in sorted(directory.glob("*.txt")):
            image = image_for_stem(directory, label.stem)
            if image is None:
                stats["aeroscan_missing_image"] += 1
                continue
            size = image_size(image)
            if size is None:
                stats["aeroscan_unreadable"] += 1
                continue
            width, height = size
            objects = []
            airframes = set()
            for line in label.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                class_id = int(parts[0])
                mapped = AEROSCAN_UAVS.get(class_id)
                if mapped is None:
                    continue
                airframe, country = mapped
                cx, cy, bw, bh = map(float, parts[1:])
                x1 = max(0.0, (cx - bw / 2) * width)
                y1 = max(0.0, (cy - bh / 2) * height)
                x2 = min(float(width), (cx + bw / 2) * width)
                y2 = min(float(height), (cy + bh / 2) * height)
                if x2 <= x1 or y2 <= y1:
                    continue
                fraction = ((x2 - x1) * (y2 - y1)) / max(1, width * height)
                objects.append((country, [x1, y1, x2, y2], 1.0,
                                int(fraction > 0.75)))
                airframes.add(airframe)
            if not objects:
                continue
            object_countries = {obj[0] for obj in objects}
            if len(object_countries) != 1:
                stats["aeroscan_mixed_country_skipped"] += 1
                continue
            hash_value = dhash_file(image)
            if hash_value is None:
                stats["aeroscan_unhashable"] += 1
                continue
            records.append({
                "image": image, "width": width, "height": height,
                "objects": objects, "dhash": hash_value,
                "stratum": Counter(o[0] for o in objects).most_common(1)[0][0],
                "meta": {
                    "provenance": "external_human_annotated",
                    "source_dataset": "AeroScan",
                    "license": "Apache-2.0",
                    "airframe": ",".join(sorted(airframes)),
                    "original_split": original_split,
                },
            })
            stats["aeroscan_positive"] += 1
            stats["aeroscan_boxes"] += len(objects)
    return records


def collect_scraped(stats):
    records = []
    for row in load_jsonl(RAW_ROOT / "manifest.jsonl"):
        image = Path(row["path"])
        xml = RAW_ROOT / "annotations" / row["airframe"] / (
            Path(row["filename"]).stem + ".xml")
        if not image.exists() or not xml.exists():
            stats["scraped_without_annotation"] += 1
            continue
        width, height, source_objects, _ = parse_voc(xml)
        objects = [(row["country"], box, score, difficult)
                   for box, score, difficult in source_objects]
        if not objects:
            stats["scraped_empty_annotation"] += 1
            continue
        hash_value = dhash_file(image)
        if hash_value is None:
            stats["scraped_unhashable"] += 1
            continue
        records.append({
            "image": image, "width": width, "height": height,
            "objects": objects, "dhash": hash_value,
            "stratum": row["country"],
            "meta": {
                "provenance": "scraped_autolabel",
                "source_dataset": row.get("source_provider") or "image_search",
                "license": row.get("license") or "source-page terms",
                "airframe": row["airframe"],
                "airframe_name": row.get("airframe_name"),
                "country": row["country"],
                "role": row.get("role"),
                "query": row.get("query"),
                "source_page": row.get("source_page"),
            },
        })
        stats["scraped_positive"] += 1
        stats["scraped_boxes"] += len(objects)
    return records


def collect_binary_negatives(stats):
    records = []
    for row in load_jsonl(BINARY_ROOT / "index.jsonl"):
        if row.get("kind") != "negative":
            continue
        split = row["split"]
        image = BINARY_ROOT / "images" / split / row["filename"]
        xml = BINARY_ROOT / "annotations" / split / (Path(row["filename"]).stem + ".xml")
        if not image.exists() or not xml.exists():
            stats["binary_negative_missing"] += 1
            continue
        width, height, _objects, xml_meta = parse_voc(xml)
        hash_value = dhash_file(image)
        if hash_value is None:
            stats["binary_negative_unhashable"] += 1
            continue
        category = row.get("category") or xml_meta.get("category") or "negative"
        records.append({
            "image": image, "width": width, "height": height,
            "objects": [], "dhash": hash_value,
            "stratum": f"negative:{category}",
            "meta": {
                "kind": "negative", "provenance": "negative",
                "category": category,
                "source_dataset": row.get("source_dataset") or xml_meta.get("source_dataset"),
                "license": row.get("license") or xml_meta.get("license"),
            },
        })
        stats["binary_negative"] += 1
    return records


def write_voc(path, filename, width, height, objects, meta):
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = meta.get("split", "")
    ET.SubElement(root, "filename").text = filename
    source = ET.SubElement(root, "source")
    ET.SubElement(source, "database").text = "drone-country-v3"
    for key, value in meta.items():
        if key != "split" and value not in (None, ""):
            ET.SubElement(source, key).text = str(value)
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"
    ET.SubElement(root, "segmented").text = "0"
    for country, box, score, difficult in objects:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = country
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = str(int(difficult))
        ET.SubElement(obj, "score").text = f"{score:.4f}"
        bb = ET.SubElement(obj, "bndbox")
        for key, value in zip(("xmin", "ymin", "xmax", "ymax"), box):
            ET.SubElement(bb, key).text = str(int(round(value)))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(minidom.parseString(ET.tostring(root)).toprettyxml(indent="  "),
                    encoding="utf-8")


def link_or_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--min-per-country", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"output already exists and is not empty: {out}")

    stats = Counter()
    records = (collect_aeroscan(stats) + collect_scraped(stats)
               + collect_binary_negatives(stats))
    if not records:
        raise SystemExit("no source records found")

    thumbs = {}

    def gray(i):
        if i not in thumbs:
            image = cv2.imread(str(records[i]["image"]), cv2.IMREAD_GRAYSCALE)
            thumbs[i] = (None if image is None else cv2.resize(
                image, (64, 64), interpolation=cv2.INTER_AREA))
        return thumbs[i]

    groups = group_duplicates(
        [r["dhash"] for r in records],
        verify=lambda i, j: ncc(gray(i), gray(j)) >= 0.92,
        chunk_rows=64,
    )
    unique, conflicts = [], []
    for group in groups:
        members = [records[i] for i in group]
        countries = {obj[0] for record in members for obj in record["objects"]}
        has_positive = any(record["objects"] for record in members)
        has_negative = any(not record["objects"] for record in members)
        if len(countries) > 1 or (has_positive and has_negative):
            conflicts.append(members)
            continue
        unique.append(max(members, key=lambda r: (
            r["width"] * r["height"], r["image"].stat().st_size)))

    per_country = Counter(r["stratum"] for r in unique if r["objects"])
    included = {country for country, count in per_country.items()
                if count >= args.min_per_country}
    dropped = {country: count for country, count in per_country.items()
               if country not in included}
    unique = [r for r in unique if not r["objects"] or r["stratum"] in included]

    rng = random.Random(args.seed)
    by_stratum = defaultdict(list)
    for record in unique:
        by_stratum[record["stratum"]].append(record)
    for stratum, members in by_stratum.items():
        rng.shuffle(members)
        n = len(members)
        n_test = max(1, round(n * 0.1)) if n >= 3 else 0
        n_val = max(1, round(n * 0.1)) if n >= 2 else 0
        while n_test + n_val >= n:
            if n_test:
                n_test -= 1
            elif n_val:
                n_val -= 1
        for index, record in enumerate(members):
            record["split"] = ("test" if index < n_test else
                               "val" if index < n_test + n_val else "train")

    for split in ("train", "val", "test"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "annotations" / split).mkdir(parents=True, exist_ok=True)

    if conflicts:
        quarantine = out / "quarantine"
        quarantine.mkdir(parents=True, exist_ok=True)
        with (quarantine / "conflicts.jsonl").open("w", encoding="utf-8") as handle:
            for group_id, members in enumerate(conflicts):
                for record in members:
                    handle.write(json.dumps({
                        "group": group_id,
                        "path": str(record["image"]),
                        "countries": sorted({o[0] for o in record["objects"]}),
                        "is_negative": not bool(record["objects"]),
                        "meta": record["meta"],
                    }) + "\n")

    index_rows = []
    used_names = set()
    for number, record in enumerate(unique):
        split = record["split"]
        base = f"{record['meta'].get('source_dataset', 'source')}_{record['image'].stem}"
        stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in base)
        if stem in used_names:
            stem = f"{stem}_{number}"
        used_names.add(stem)
        filename = stem + record["image"].suffix.lower()
        link_or_copy(record["image"], out / "images" / split / filename)
        meta = dict(record["meta"], split=split)
        write_voc(out / "annotations" / split / f"{stem}.xml", filename,
                  record["width"], record["height"], record["objects"], meta)
        index_rows.append({
            "filename": filename, "split": split,
            "kind": "drone" if record["objects"] else "negative",
            "class_name": record["stratum"] if record["objects"] else None,
            "dhash": record["dhash"].tobytes().hex(), **record["meta"],
        })

    label_map = {country: i + 1 for i, country in enumerate(sorted(included))}
    (out / "classes.json").write_text(json.dumps({
        "scheme": "country_of_origin",
        "definition": "Airframe design/manufacturing origin, not current operator",
        "label_map": label_map,
        "sources": {
            "AeroScan": {"license": "Apache-2.0"},
            "scraped_autolabel": {"licenses": "per-image manifest"},
            "binary_v3_hard_negatives": {"licenses": "per-image annotation"},
        },
    }, indent=2), encoding="utf-8")
    with (out / "index.jsonl").open("w", encoding="utf-8") as handle:
        for row in index_rows:
            handle.write(json.dumps(row) + "\n")

    split_counts = Counter(r["split"] for r in unique)
    kind_counts = Counter("positive" if r["objects"] else "negative" for r in unique)
    final_country = Counter(r["stratum"] for r in unique if r["objects"])
    report = [
        "Country-of-origin drone dataset v3", "==================================", "",
        f"total images: {len(unique)}",
        *(f"{split}: {split_counts[split]}" for split in ("train", "val", "test")),
        f"positive images: {kind_counts['positive']}",
        f"negative images: {kind_counts['negative']}",
        f"countries: {len(label_map)}", "", "country balance:",
        *(f"  {country}: {count}" for country, count in final_country.most_common()),
        "", f"duplicates removed: {len(records) - len(groups)}",
        f"conflict groups quarantined: {len(conflicts)}",
        f"countries below threshold: {json.dumps(dropped, sort_keys=True)}",
        "", "source counters:",
        *(f"  {key}: {value}" for key, value in sorted(stats.items())),
    ]
    (out / "BUILD_REPORT.txt").write_text("\n".join(report) + "\n",
                                           encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
