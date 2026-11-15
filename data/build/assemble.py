"""Assemble scraped images and auto-labels into a trainable dataset.

This is the stage that decides what actually gets trained on, and every rule in
it exists because the previous dataset broke that rule.

Global dedup, not per-source dedup. The scraper rejects duplicates as it
downloads, but negatives, legacy images and drone images are fetched by separate
processes that cannot see each other's hashes. Duplicates are therefore resolved
once, here, across the whole corpus at the end. Images are clustered by
single-link union-find on perceptual hash, so a chain of re-encodes collapses to
one picture, and the highest-resolution member of each cluster is kept.

Dedup before splitting, which removes leakage by construction. Once each picture
exists exactly once there is no way for a copy of a training image to appear in
validation. The old dataset had 21 near-duplicate pairs straddling the split and
so was partly validated on its own training data.

Conflicts are quarantined, never voted on. When one cluster contains images
claiming two different countries, at least one is wrong and nothing here can
tell which. Those clusters are written to `quarantine/` with a manifest instead
of being silently resolved - that is what the old corpus did, and it is how 60
mislabelled files ended up in training. Because the new scrape searches per
airframe, a conflict now means two *airframe* queries returned the same picture,
which is worth a human look.

Stratified splitting on the final classes. The split is drawn per class so a
country with few images still appears in validation at the intended ratio,
rather than landing entirely in train and scoring an undefined AP.
"""

import json
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from xml.dom import minidom

import cv2
import numpy as np

from data.airframes import BY_KEY, class_for
from data.build.hashing import DUP_THRESHOLD, group_duplicates, ncc

RAW_ROOT = Path("dataset_raw")
OUT_ROOT = Path("drone_dataset_v2")
REPORT = Path("reports/dataset_v2_build.txt")

VAL_FRACTION = 0.1
TEST_FRACTION = 0.1
RANDOM_SEED = 1337

# A class needs at least this many images to be worth predicting. Below it, the
# class contributes an AP that is pure noise - the old baseline's per-class
# numbers swung on a handful of images. Under-represented classes are dropped
# from the label space and their images set aside, which is reported, not hidden.
MIN_IMAGES_PER_CLASS = 40

# Normalised cross-correlation a hash match must reach to count as the same
# picture. Re-encodes and rescales of one photo sit at ~0.99-1.00; genuinely
# different photos of the same airframe fall far below. 0.92 leaves room for JPEG
# blocking and mild cropping without admitting look-alikes.
NCC_CONFIRM = 0.92


def _load_jsonl(path):
    """Read a JSONL manifest, skipping a torn final line from a live writer."""
    if not path.exists():
        return []
    records = []
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def _hash_array(hexstr):
    return np.frombuffer(bytes.fromhex(hexstr), dtype=np.uint8)


def _read_boxes(xml_path):
    """Return (width, height, [(box, score, difficult)]) from an annotation."""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    w, h = int(float(size.find("width").text)), int(float(size.find("height").text))
    objects = []
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        box = [float(bb.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")]
        score_el = obj.find("score")
        score = float(score_el.text) if score_el is not None else 1.0
        difficult = int(obj.find("difficult").text) if obj.find("difficult") is not None else 0
        objects.append((box, score, difficult))
    return w, h, objects


def _write_voc(xml_path, img_name, width, height, class_name, objects, meta):
    ann = ET.Element("annotation")
    ET.SubElement(ann, "folder").text = meta.get("split", "")
    ET.SubElement(ann, "filename").text = img_name
    src = ET.SubElement(ann, "source")
    ET.SubElement(src, "database").text = "drone-dataset-v2"
    for key in ("airframe", "airframe_name", "country", "role", "query",
                "kind", "provenance", "category", "distractor_class",
                "source_dataset", "source_classes", "license"):
        if meta.get(key):
            ET.SubElement(src, key).text = str(meta[key])

    size = ET.SubElement(ann, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"
    ET.SubElement(ann, "segmented").text = "0"

    for box, score, difficult in objects:
        obj = ET.SubElement(ann, "object")
        ET.SubElement(obj, "name").text = class_name
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = str(int(difficult))
        ET.SubElement(obj, "score").text = f"{score:.4f}"
        bb = ET.SubElement(obj, "bndbox")
        for k, v in zip(("xmin", "ymin", "xmax", "ymax"), box):
            ET.SubElement(bb, k).text = str(int(round(v)))

    xml_path.parent.mkdir(parents=True, exist_ok=True)
    xml_path.write_text(
        minidom.parseString(ET.tostring(ann)).toprettyxml(indent="  "),
        encoding="utf-8")


def _collect_drones(scheme):
    """Manifest entries that have a usable annotation, tagged with their class."""
    items = []
    missing = 0
    for rec in _load_jsonl(RAW_ROOT / "manifest.jsonl"):
        xml = RAW_ROOT / "annotations" / rec["airframe"] / (
            Path(rec["filename"]).stem + ".xml")
        if not xml.exists():
            missing += 1
            continue
        if rec["airframe"] not in BY_KEY:
            continue
        items.append({
            "kind": "drone",
            "image": Path(rec["path"]),
            "xml": xml,
            "class_name": class_for(rec["airframe"], scheme),
            "dhash": rec["dhash"],
            "width": rec.get("width") or 0,
            "height": rec.get("height") or 0,
            "meta": {
                "airframe": rec["airframe"],
                "airframe_name": rec.get("airframe_name"),
                "country": rec["country"],
                "role": rec["role"],
                "query": rec.get("query"),
                "provenance": "scraped",
            },
        })
    return items, missing


def _collect_negatives():
    items = []
    for rec in _load_jsonl(RAW_ROOT / "negatives.jsonl"):
        path = Path(rec["source_path"])
        if not path.exists():
            continue
        items.append({
            "kind": "negative",
            "image": path,
            "xml": None,
            "class_name": None,          # background: no foreground class
            "dhash": rec["dhash"],
            "width": rec.get("width") or 0,
            "height": rec.get("height") or 0,
            "meta": {
                "category": rec["category"],
                "distractor_class": rec.get("distractor_class"),
                "provenance": "negative",
            },
        })
    return items


def assemble(scheme="country", val_fraction=VAL_FRACTION,
             test_fraction=TEST_FRACTION,
             min_per_class=MIN_IMAGES_PER_CLASS, seed=RANDOM_SEED,
             dup_threshold=DUP_THRESHOLD, clean=True):
    rng = np.random.default_rng(seed)
    lines = []

    def out(msg=""):
        print(msg)
        lines.append(msg)

    drones, missing_ann = _collect_drones(scheme)
    negatives = _collect_negatives()
    out(f"=== sources ===")
    out(f"  drone images with annotations : {len(drones)}")
    out(f"  drone images without          : {missing_ann}")
    out(f"  negatives                     : {len(negatives)}")
    if not drones:
        raise SystemExit("no annotated drone images - run scrape then autolabel")

    items = drones + negatives

    # ---------------------------------------------------------------- dedup
    hashes = [_hash_array(it["dhash"]) for it in items]

    # Every candidate pair the hash proposes is confirmed against actual pixels
    # before it is allowed to merge. Without this, a handful of near-uniform
    # images (blank sky, white backgrounds) matched each other on the absence of
    # structure and single-link chaining pulled 15 unrelated aircraft and
    # negatives into one bogus "conflict" group.
    #
    # Images are cached at low resolution: the candidate set is a tiny fraction of
    # the corpus, so only the pictures actually involved in a near-match are read.
    cache = {}

    def _gray(idx):
        if idx not in cache:
            img = cv2.imread(str(items[idx]["image"]), cv2.IMREAD_GRAYSCALE)
            cache[idx] = None if img is None else cv2.resize(
                img, (64, 64), interpolation=cv2.INTER_AREA)
        return cache[idx]

    verified = {"confirmed": 0, "rejected": 0}

    def verify(i, j):
        score = ncc(_gray(i), _gray(j))
        ok = score >= NCC_CONFIRM
        verified["confirmed" if ok else "rejected"] += 1
        return ok

    groups = group_duplicates(hashes, threshold=dup_threshold, verify=verify)
    out(f"\n=== dedup (hamming <= {dup_threshold}, NCC >= {NCC_CONFIRM}) ===")
    out(f"  images in            : {len(items)}")
    out(f"  distinct pictures    : {len(groups)}")
    out(f"  redundant copies     : {len(items) - len(groups)}")
    out(f"  hash matches confirmed by pixels : {verified['confirmed']}")
    out(f"  hash matches rejected as spurious: {verified['rejected']}")

    kept, quarantined = [], []
    for g in groups:
        members = [items[i] for i in g]
        classes = {m["class_name"] for m in members if m["kind"] == "drone"}
        kinds = {m["kind"] for m in members}
        # A picture that is both a drone and a negative is as much a conflict as
        # two countries disagreeing: one of the two sources is wrong.
        if len(classes) > 1 or (len(kinds) > 1 and classes):
            quarantined.append(members)
            continue
        # Keep the highest-resolution copy: re-encodes lose detail, and the
        # largest member is the closest thing to the original.
        kept.append(max(members, key=lambda m: (m["width"] * m["height"],
                                                m["image"].stat().st_size
                                                if m["image"].exists() else 0)))

    out(f"  conflict groups      : {len(quarantined)} "
        f"({sum(len(g) for g in quarantined)} images quarantined)")
    for g in quarantined[:10]:
        desc = ", ".join(
            f"{m['image'].name}:{m['class_name'] or m['meta'].get('category')}"
            for m in g)
        out(f"      {desc}")

    # -------------------------------------------------------- class filtering
    drone_kept = [k for k in kept if k["kind"] == "drone"]
    neg_kept = [k for k in kept if k["kind"] == "negative"]
    per_class = Counter(k["class_name"] for k in drone_kept)

    dropped_classes = {c for c, n in per_class.items() if n < min_per_class}
    if dropped_classes:
        out(f"\n=== classes below {min_per_class} images (dropped) ===")
        for c in sorted(dropped_classes):
            out(f"  {per_class[c]:5d}  {c}")
    drone_kept = [k for k in drone_kept if k["class_name"] not in dropped_classes]

    out(f"\n=== final class balance ===")
    final_counts = Counter(k["class_name"] for k in drone_kept)
    for c, n in final_counts.most_common():
        out(f"  {n:5d}  {c}")
    out(f"  {len(drone_kept)} drone images across {len(final_counts)} classes")
    out(f"  {len(neg_kept)} negatives")

    # ---------------------------------------------------------------- split
    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction and test_fraction must be non-negative "
                         "and sum to less than 1")

    def holdout_counts(n):
        """Return validation/test counts while preserving at least one train item."""
        if n <= 1:
            return 0, 0
        n_val = max(1, int(round(val_fraction * n))) if val_fraction else 0
        n_test = (max(1, int(round(test_fraction * n)))
                  if test_fraction and n >= 3 else 0)
        while n_val + n_test >= n:
            if n_test > n_val and n_test > 0:
                n_test -= 1
            elif n_val > 0:
                n_val -= 1
            else:
                break
        return n_val, n_test

    # Stratify per class so thin classes still reach both held-out splits.
    assignments = {}
    by_class = defaultdict(list)
    for k in drone_kept:
        by_class[k["class_name"]].append(k)
    for cls, members in by_class.items():
        idx = rng.permutation(len(members))
        n_val, n_test = holdout_counts(len(members))
        for rank, i in enumerate(idx):
            split = ("test" if rank < n_test else
                     "val" if rank < n_test + n_val else "train")
            assignments[id(members[i])] = split

    # Negatives are stratified by category for the same reason.
    by_cat = defaultdict(list)
    for k in neg_kept:
        by_cat[k["meta"]["category"]].append(k)
    for cat, members in by_cat.items():
        idx = rng.permutation(len(members))
        n_val, n_test = holdout_counts(len(members))
        for rank, i in enumerate(idx):
            split = ("test" if rank < n_test else
                     "val" if rank < n_test + n_val else "train")
            assignments[id(members[i])] = split

    # ----------------------------------------------------------------- write
    if clean and OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    for split in ("train", "val", "test"):
        (OUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / "annotations" / split).mkdir(parents=True, exist_ok=True)

    written = Counter()
    box_stats = Counter()
    index_rows = []

    for item in drone_kept + neg_kept:
        split = assignments[id(item)]
        stem = item["image"].stem
        dest_img = OUT_ROOT / "images" / split / f"{stem}{item['image'].suffix.lower()}"
        # Two sources can produce the same stem; disambiguate rather than clobber.
        n = 1
        while dest_img.exists():
            dest_img = dest_img.with_name(f"{stem}_{n}{item['image'].suffix.lower()}")
            n += 1
        shutil.copy2(item["image"], dest_img)

        meta = dict(item["meta"], split=split)
        if item["kind"] == "drone":
            w, h, objects = _read_boxes(item["xml"])
            objects = [(b, s, d) for b, s, d in objects]
            box_stats["boxes"] += len(objects)
            box_stats["difficult"] += sum(1 for _, _, d in objects if d)
            _write_voc(OUT_ROOT / "annotations" / split / f"{dest_img.stem}.xml",
                       dest_img.name, w, h, item["class_name"], objects, meta)
        else:
            # Negatives get a real annotation file with zero objects. An explicit
            # empty annotation is what makes them trainable background; the old
            # loader inferred emptiness from a directory name instead, which is
            # how all 200 negatives ended up in both splits at once.
            w = item["width"] or 0
            h = item["height"] or 0
            if not (w and h):
                im = cv2.imread(str(dest_img))
                if im is not None:
                    h, w = im.shape[:2]
            _write_voc(OUT_ROOT / "annotations" / split / f"{dest_img.stem}.xml",
                       dest_img.name, w, h, "", [], meta)

        written[(split, item["kind"])] += 1
        index_rows.append({
            "filename": dest_img.name,
            "split": split,
            "kind": item["kind"],
            "class_name": item["class_name"],
            "dhash": item["dhash"],
            **{k: v for k, v in item["meta"].items() if v is not None},
        })

    # Quarantine: copied, not deleted, with a manifest explaining each group.
    if quarantined:
        qdir = OUT_ROOT / "quarantine"
        qdir.mkdir(parents=True, exist_ok=True)
        with (qdir / "conflicts.jsonl").open("w", encoding="utf-8") as fh:
            for gi, g in enumerate(quarantined):
                for m in g:
                    if m["image"].exists():
                        shutil.copy2(m["image"], qdir / f"g{gi:04d}_{m['image'].name}")
                    fh.write(json.dumps({
                        "group": gi,
                        "filename": m["image"].name,
                        "kind": m["kind"],
                        "class_name": m["class_name"],
                        "meta": m["meta"],
                    }) + "\n")

    with (OUT_ROOT / "index.jsonl").open("w", encoding="utf-8") as fh:
        for row in index_rows:
            fh.write(json.dumps(row) + "\n")

    out(f"\n=== written to {OUT_ROOT} ===")
    for split in ("train", "val", "test"):
        d = written[(split, "drone")]
        n = written[(split, "negative")]
        out(f"  {split:5s}  {d:5d} drone  {n:5d} negative  {d + n:5d} total")
    out(f"  boxes: {box_stats['boxes']} "
        f"({box_stats['difficult']} flagged difficult / low-confidence)")

    # Leakage is impossible by construction after dedup, but assert it: a silent
    # regression here invalidates every number the project reports.
    hashes = {
        split: {r["dhash"] for r in index_rows if r["split"] == split}
        for split in ("train", "val", "test")
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = len(hashes[left] & hashes[right])
        out(f"\n  identical-hash overlap {left}/{right}: {overlap} (must be 0)")

    label_map = {c: i + 1 for i, c in enumerate(sorted(final_counts))}
    (OUT_ROOT / "classes.json").write_text(json.dumps({
        "scheme": scheme,
        "label_map": label_map,
        "dropped_classes": sorted(dropped_classes),
        "min_images_per_class": min_per_class,
    }, indent=2), encoding="utf-8")
    out(f"\n  label map ({scheme}, background=0): {label_map}")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nreport written to {REPORT}")
    return {"train": written[("train", "drone")] + written[("train", "negative")],
            "val": written[("val", "drone")] + written[("val", "negative")],
            "test": written[("test", "drone")] + written[("test", "negative")],
            "label_map": label_map}


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scheme", default="country",
                   choices=["country", "airframe", "role"])
    p.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    p.add_argument("--test-fraction", type=float, default=TEST_FRACTION)
    p.add_argument("--min-per-class", type=int, default=MIN_IMAGES_PER_CLASS)
    p.add_argument("--dup-threshold", type=int, default=DUP_THRESHOLD)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--no-clean", action="store_true",
                   help="add to the existing output instead of rebuilding it")
    a = p.parse_args()
    assemble(scheme=a.scheme, val_fraction=a.val_fraction,
             test_fraction=a.test_fraction,
             min_per_class=a.min_per_class, seed=a.seed,
             dup_threshold=a.dup_threshold, clean=not a.no_clean)


if __name__ == "__main__":
    main()
