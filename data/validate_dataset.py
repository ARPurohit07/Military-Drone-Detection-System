"""Pre-training dataset checks.

Catches the annotation problems that quietly degrade a detector - boxes that
cover the whole frame, degenerate or out-of-bounds boxes, images with no
annotation, unknown class names - before you spend hours training on them.

    python -m data.validate_dataset
    python -m data.validate_dataset --root drone_dataset_country_v3 --split train

Two checks are specific to the rebuilt dataset.

Intentional negatives are not errors. Hard negatives are stored as annotations
containing zero objects, so a blanket "empty annotation" warning would flag every
one of them and bury the real problems. An empty annotation whose provenance says
it is a negative is counted, not flagged; an empty annotation without that marker
still is, because that is a drone image whose auto-labelling produced nothing.

Cross-split duplication is checked, not assumed. The builder deduplicates before
splitting, which makes leakage impossible in principle - but that is exactly the
kind of invariant that breaks silently and invalidates every metric downstream.
Perceptual hashes are recomputed here from the images on disk and compared across
splits, so the guarantee is verified against the files rather than trusted.
"""

import argparse
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import cv2

from config import CLASS_MAP, TrainConfig, load_label_map

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_annotation(xml_path):
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    width = float(size.find("width").text)
    height = float(size.find("height").text)

    meta = {}
    source = root.find("source")
    if source is not None:
        for child in source:
            if child.text:
                meta[child.tag] = child.text

    objects = []
    for obj in root.findall("object"):
        name_el = obj.find("name")
        name = (name_el.text or "").strip() if name_el is not None else ""
        difficult_el = obj.find("difficult")
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        objects.append({
            "name": name,
            "difficult": int(difficult_el.text) if difficult_el is not None
            and difficult_el.text else 0,
            "box": tuple(float(bnd.find(k).text)
                         for k in ("xmin", "ymin", "xmax", "ymax")),
        })
    return width, height, objects, meta


def check_split(root, split, max_area_frac):
    root = Path(root)
    img_dir = root / "images" / split
    ann_dir = root / "annotations" / split
    if not ann_dir.exists():
        raise SystemExit(f"no annotations at {ann_dir}")

    issues = {
        "huge_box": [], "degenerate": [], "out_of_bounds": [],
        "unknown_class": [], "no_annotation": [], "empty_annotation": [],
        "duplicate_stem": [], "missing_image": [],
    }
    class_counts = Counter()
    stats = Counter()

    ann_stems = {p.stem for p in ann_dir.glob("*.xml")}

    # An image whose stem exists in several formats: the loader takes the first
    # extension it finds and silently ignores the rest.
    by_stem = defaultdict(list)
    if img_dir.exists():
        for img in img_dir.iterdir():
            if img.suffix.lower() in IMAGE_EXTENSIONS:
                by_stem[img.stem].append(img.suffix)
    for stem, exts in sorted(by_stem.items()):
        if len(exts) > 1:
            issues["duplicate_stem"].append(f"{stem} {sorted(exts)}")
        if stem not in ann_stems:
            issues["no_annotation"].append(stem)

    for xml_path in sorted(ann_dir.glob("*.xml")):
        if xml_path.stem not in by_stem:
            issues["missing_image"].append(xml_path.stem)
            continue

        width, height, objects, meta = parse_annotation(xml_path)
        named = [o for o in objects if o["name"]]
        kept = [o for o in named if o["difficult"] != 1]
        stats["difficult_boxes"] += len(named) - len(kept)

        is_negative = (meta.get("provenance") == "negative"
                       or bool(meta.get("category")))
        if not named:
            if is_negative:
                stats["negatives"] += 1
            else:
                # A drone image the auto-labeller found nothing in: it will be
                # loaded as background, teaching the model that this airframe is
                # not a drone. Worth flagging.
                issues["empty_annotation"].append(
                    f"{xml_path.stem} (airframe={meta.get('airframe', '?')})")
            continue

        stats["positives"] += 1
        for obj in kept:
            name = obj["name"]
            class_counts[name] += 1
            if name not in CLASS_MAP:
                issues["unknown_class"].append(f"{xml_path.stem}: {name!r}")

            x1, y1, x2, y2 = obj["box"]
            if x2 <= x1 or y2 <= y1:
                issues["degenerate"].append(f"{xml_path.stem}: {obj['box']}")
                continue
            if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                issues["out_of_bounds"].append(
                    f"{xml_path.stem}: {obj['box']} in {width:.0f}x{height:.0f}")

            frac = ((x2 - x1) * (y2 - y1)) / (width * height)
            if frac > max_area_frac:
                issues["huge_box"].append(f"{xml_path.stem}: {frac:.1%} of frame")

    return issues, class_counts, stats, len(ann_stems)


def check_leakage(root, splits, threshold=6, ncc_confirm=0.92):
    """Recompute perceptual hashes from disk and report pictures spanning splits.

    Deliberately independent of the builder: it re-derives hashes from the written
    files rather than trusting the manifests, because this is the check that says
    whether validation is honest, and an invariant verified only against the
    bookkeeping that produced it is not verified at all.

    Two refinements over a plain hash comparison, both for the same reason - a
    false leakage report is as damaging as a missed one, since it would send
    someone looking for a problem that is not there:

    Hashes with too little structure are skipped. Blank sky and white-background
    images have almost no bits set and match each other by default.

    Every hash match is confirmed on pixels by normalised cross-correlation, so a
    coincidence is not reported as leakage.

    Comparison is vectorised in blocks; the pairwise Python loop it replaces ran
    train x val (~21 million) iterations.
    """
    import numpy as np

    from data.build.hashing import dhash_file, is_informative, ncc

    per_split = {}
    for split in splits:
        img_dir = Path(root) / "images" / split
        if not img_dir.exists():
            continue
        entries = []
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            h = dhash_file(img)
            if h is not None and is_informative(h):
                entries.append((img.name, img, h))
        per_split[split] = entries

    if len(per_split) < 2:
        return None

    (a_name, a), (b_name, b) = list(per_split.items())[:2]
    if not a or not b:
        return {"splits": (a_name, b_name), "counts": (len(a), len(b)),
                "collisions": [], "rejected": 0}

    bits_a = np.unpackbits(np.array([h for _, _, h in a], dtype=np.uint8), axis=1)
    bits_b = np.unpackbits(np.array([h for _, _, h in b], dtype=np.uint8), axis=1)

    collisions, rejected = [], 0
    cache = {}

    def gray(path):
        if path not in cache:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            cache[path] = None if img is None else cv2.resize(
                img, (64, 64), interpolation=cv2.INTER_AREA)
        return cache[path]

    for start in range(0, len(bits_a), 512):
        stop = min(start + 512, len(bits_a))
        block = (bits_a[start:stop, None, :] != bits_b[None, :, :]).sum(axis=2)
        for r, c in zip(*np.where(block <= threshold)):
            ia, ib = start + int(r), int(c)
            if ncc(gray(a[ia][1]), gray(b[ib][1])) >= ncc_confirm:
                collisions.append((a[ia][0], b[ib][0], int(block[r, c])))
            else:
                rejected += 1

    return {
        "splits": (a_name, b_name),
        "counts": (len(a), len(b)),
        "collisions": collisions,
        "rejected": rejected,
    }


DESCRIPTIONS = {
    "huge_box": "boxes covering most of the frame (teaches 'drone' = everything)",
    "degenerate": "zero/negative-area boxes",
    "out_of_bounds": "boxes extending past the image edge",
    "unknown_class": "class names absent from CLASS_MAP (these raise at load)",
    "no_annotation": "images with no matching annotation (silently unused)",
    "empty_annotation": "drone images whose annotation has no objects",
    "duplicate_stem": "same stem in multiple formats (only one is loaded)",
    "missing_image": "annotations with no matching image",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None)
    parser.add_argument("--split", default=None,
                        help="Default: every available train/val/test split.")
    parser.add_argument("--max-area", type=float, default=0.75,
                        help="Flag boxes covering more than this fraction.")
    parser.add_argument("--show", type=int, default=8, help="Examples per issue.")
    parser.add_argument("--check-leakage", action="store_true",
                        help="recompute hashes and verify no picture spans splits")
    args = parser.parse_args()

    root = args.root or TrainConfig().root
    # Validation must use the dataset being checked, not config.py's default
    # root. Binary and country datasets intentionally have different maps.
    active_map, _, source = load_label_map(root)
    globals()["CLASS_MAP"] = active_map
    splits = ([args.split] if args.split else
              [s for s in ("train", "val", "test")
               if (Path(root) / "annotations" / s).exists()])
    print(f"dataset: {root}")
    print(f"label map: {source}")

    total_problems = 0
    for split in splits:
        issues, class_counts, stats, n_ann = check_split(
            root, split, args.max_area)
        print(f"\n=== {split} ({n_ann} annotations) ===")
        print(f"  {stats['positives']} with objects, "
              f"{stats['negatives']} negatives, "
              f"{stats['difficult_boxes']} boxes flagged difficult")
        print("Class distribution:")
        for name, count in class_counts.most_common():
            mark = "" if name in CLASS_MAP else "  <-- UNKNOWN"
            print(f"  {name:<18} {count:>5}{mark}")

        for key, items in issues.items():
            if not items:
                continue
            total_problems += len(items)
            print(f"\n{len(items)} x {key} - {DESCRIPTIONS[key]}")
            for item in items[:args.show]:
                print(f"    {item}")
            if len(items) > args.show:
                print(f"    ... and {len(items) - args.show} more")

    if args.check_leakage and len(splits) > 1:
        from itertools import combinations

        print(f"\n=== cross-split duplication ===")
        for pair in combinations(splits, 2):
            result = check_leakage(root, pair)
            if result is None:
                continue
            a, b = result["splits"]
            na, nb = result["counts"]
            n = len(result["collisions"])
            print(f"  compared {na} {a} images against {nb} {b} images")
            print(f"  near-duplicate pairs spanning splits: {n} (must be 0)")
            if result.get("rejected"):
                print(f"  hash matches rejected as spurious by pixel check: "
                      f"{result['rejected']}")
            for name_a, name_b, d in result["collisions"][:args.show]:
                print(f"    {name_a}  <->  {name_b}   (hamming {d})")
            total_problems += n

    print(f"\n{'-' * 50}")
    print(f"Total flagged: {total_problems}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
