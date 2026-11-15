"""Collect hard negatives - the images that teach the model what is *not* a drone.

Two sources feed this stage. The existing `Negative/` directory already holds
scraped imagery, and more can be fetched for any category that is thin.

The existing pool needs care, because its size is an illusion: 23,940 files
there collapse to **380 distinct pictures**. The rest are offline augmentation
chains (`birds_in_sky_0_aug0_aug0_aug0.jpg` and so on, up to five levels deep).
Feeding those to a detector inflates the epoch count without adding information,
and because the chains were generated before any train/val split existed,
augmented copies of one picture would land on both sides of it - the model would
be validated on transformed versions of its own training images. So chains are
stripped back to their originals here, and augmentation is applied on the fly at
training time instead, where it is re-rolled every epoch and cannot leak.

Categories are split by what they can teach:

PURE_BACKGROUND - sky, trees, rooms, hands. Nothing drone-like is present, so
these can only ever be empty-annotation background images. Their whole job is to
push down the false-positive rate.

DISTRACTORS - birds, airliners, helicopters, balloons, kites, insects. These
contain a small flying object that a drone detector genuinely confuses for its
target. They are loaded as background *now*, which is what suppresses false
positives, but they are tracked separately and their category is recorded in
each annotation file, so promoting them to predicted classes later needs boxes
drawn on them and nothing else - no rebuild, and no renumbering of drone labels.
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2

from data.airframes import DISTRACTORS, PURE_BACKGROUND
from data.build.hashing import DuplicateIndex, dhash_file

RAW_ROOT = Path("dataset_raw")
NEG_MANIFEST = RAW_ROOT / "negatives.jsonl"

# Augmentation chains look like `<stem>_aug0`, `<stem>_aug0_aug1`, ... The suffix
# repeats, so strip every trailing occurrence to recover the original stem.
AUG_SUFFIX = re.compile(r"(_aug\d+)+$")

# Category -> the search query that finds more of it, for topping up thin
# categories. Keys match the prefixes already used in `Negative/`.
NEGATIVE_QUERIES = {
    "birds_in_sky": "birds flying in sky",
    "planes_far": "airplane far away in sky",
    "helicopters_far": "helicopter flying distant sky",
    "balloons_in_sky": "hot air balloon in sky",
    "kites_flying": "kite flying in sky",
    "insects_flying": "insect flying close up sky",
    "floating_plastic_bag": "plastic bag floating in wind",
    "sky": "clear blue sky",
    "sky_clouds": "clouds in sky",
    "landscape": "landscape scenery horizon",
    "trees": "trees from below sky",
    "branches_sky": "bare tree branches against sky",
    "leaves_in_sky": "leaves falling against sky",
    "people": "people outdoors crowd",
    "room": "indoor room interior",
    "windows": "window from inside",
    "curtains": "curtains window indoor",
    "hands": "human hand close up",
    "fingers": "fingers pointing close up",
    # Categories absent from the original pool. Street furniture and wires are
    # the classic sources of false positives for an airborne detector looking at
    # a cluttered skyline.
    "streetlights": "street light against sky",
    "power_lines": "power lines against sky",
    "antennas": "rooftop antenna against sky",
    "wind_turbines": "wind turbine against sky",
    "fireworks": "fireworks in night sky",
    "night_sky": "night sky stars",
    "seagulls": "seagull flying close up",
    "bats_flying": "bat flying in sky",
    "paragliders": "paraglider in sky",
    "model_aircraft": "rc model airplane flying",
    # Deployment hard negatives discovered from the 517 captured camera frames.
    # These suppress the high-confidence phone/person false positives without
    # changing the semantic target: a physical military drone in the scene.
    "drone_on_screen": "military drone photo displayed on smartphone screen",
    "phone_closeup": "person holding smartphone close to camera",
    "indoor_person_closeup": "person indoors close to webcam",
}


# A single query string returns a largely fixed result set - one query per
# category was topping out around 70 unique images, well short of the target.
# These extra phrasings per category are the same trick used for airframes in
# data/airframes.py, and they matter more here than they look: the false-positive
# rate is measured per category, so a thin category produces a rate computed over
# 13 images, which is noise.
EXTRA_NEGATIVE_QUERIES = {
    "birds_in_sky": ("flock of birds flying", "bird silhouette against sky",
                     "single bird soaring"),
    "planes_far": ("airliner in flight distant", "jet plane against clouds",
                   "aircraft silhouette sky"),
    "helicopters_far": ("military helicopter in flight",
                        "helicopter silhouette sky"),
    "balloons_in_sky": ("weather balloon sky", "party balloons floating sky"),
    "kites_flying": ("kite festival sky", "child flying kite"),
    "insects_flying": ("dragonfly in flight", "bee flying close up"),
    "floating_plastic_bag": ("debris floating in air", "litter blowing wind"),
    "sky": ("blue sky no clouds", "overcast grey sky"),
    "sky_clouds": ("cumulus clouds", "dramatic storm clouds"),
    "landscape": ("countryside horizon view", "mountain landscape sky"),
    "trees": ("tree canopy from below", "forest against sky"),
    "branches_sky": ("winter branches silhouette", "twigs against grey sky"),
    "leaves_in_sky": ("autumn leaves falling", "leaf blowing in wind"),
    "people": ("crowd of people outdoors", "person standing outdoors"),
    "room": ("living room interior", "empty office interior"),
    "windows": ("window looking outside", "glass window building"),
    "curtains": ("window curtains daylight",),
    "hands": ("open palm close up",),
    "fingers": ("pointing finger sky",),
    "streetlights": ("lamp post against sky", "street lamp silhouette"),
    "power_lines": ("electricity pylon sky", "telephone wires against sky"),
    "antennas": ("tv aerial on roof", "radio mast against sky"),
    "wind_turbines": ("wind farm turbines sky",),
    "fireworks": ("fireworks display night",),
    "night_sky": ("stars in night sky", "moon in dark sky"),
    "seagulls": ("seagull in flight", "gulls flying over sea"),
    "bats_flying": ("bats at dusk sky",),
    "paragliders": ("paraglider flying mountain", "hang glider in sky"),
    "model_aircraft": ("rc plane in flight", "model glider flying"),
    "drone_on_screen": ("drone image on phone screen",
                        "military aircraft photo on computer monitor"),
    "phone_closeup": ("hand holding phone close up", "smartphone in hand indoors"),
    "indoor_person_closeup": ("person sitting in room webcam",
                              "person close to laptop camera"),
}


def queries_for(category):
    """All search phrasings for a negative category, primary first."""
    primary = NEGATIVE_QUERIES.get(category)
    if not primary:
        return ()
    return (primary,) + tuple(EXTRA_NEGATIVE_QUERIES.get(category, ()))


def category_of(stem):
    """Recover the category from a filename stem.

    `birds_in_sky_12` -> `birds_in_sky`, and `neg_birds_in_sky_0000` likewise, so
    that a directory this module wrote itself can be re-indexed by `harvest` -
    which is the recovery path if a scrape is interrupted and its manifest lines
    need rebuilding from the files on disk. No category name contains a digit, so
    stripping the trailing index is unambiguous.
    """
    base = AUG_SUFFIX.sub("", stem)
    base = re.sub(r"^neg_", "", base)
    return re.sub(r"_\d+$", "", base)


def kind_of(category):
    """Return 'distractor', 'background', or None for an unknown category."""
    if category in DISTRACTORS:
        return "distractor"
    for distractor, prefixes in DISTRACTORS.items():
        if category in prefixes:
            return "distractor"
    if category in PURE_BACKGROUND:
        return "background"
    return None


def distractor_class(category):
    """Map a negative category to its future distractor class, if any."""
    for distractor, prefixes in DISTRACTORS.items():
        if category == distractor or category in prefixes:
            return distractor
    return None


def harvest(source_dirs=("Negative",), min_side=120):
    """Index unique, non-augmented negatives from existing directories.

    Returns the list of manifest records written.
    """
    index = DuplicateIndex()
    records = []
    counters = defaultdict(int)
    skipped = Counter()

    candidates = []
    for src in source_dirs:
        root = Path(src)
        if not root.exists():
            print(f"  (no such directory: {src})")
            continue
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            if AUG_SUFFIX.search(path.stem):
                skipped["augmented_copy"] += 1
                continue
            candidates.append(path)

    print(f"scanning {len(candidates)} non-augmented candidates "
          f"({skipped['augmented_copy']} augmentation copies skipped)")

    for path in candidates:
        # `detections/` contains frames captured by the previous deployment
        # whenever that model fired. They contain no physical airborne drone;
        # many show a phone displaying a drone photo, while others are people or
        # room clutter. Treat the entire incident folder as one mined hard-
        # negative category and let global perceptual dedup remove adjacent
        # near-identical video frames.
        category = ("deployment_false_positive"
                    if path.parent.name.lower() == "detections"
                    else category_of(path.stem))
        kind = kind_of(category)
        if kind is None:
            skipped["unknown_category"] += 1
            continue

        img = cv2.imread(str(path))
        if img is None:
            skipped["unreadable"] += 1
            continue
        h, w = img.shape[:2]
        if min(h, w) < min_side:
            skipped["too_small"] += 1
            continue

        hsh = dhash_file(path)
        if index.find(hsh) is not None:
            skipped["duplicate"] += 1
            continue
        index.add(hsh, path.name)

        idx = counters[category]
        counters[category] += 1
        records.append({
            "filename": f"neg_{category}_{idx:04d}{path.suffix.lower()}",
            "source_path": str(path.as_posix()),
            "category": category,
            "kind": kind,
            "distractor_class": distractor_class(category),
            "width": w,
            "height": h,
            "dhash": hsh.tobytes().hex(),
        })

    NEG_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with NEG_MANIFEST.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")

    print(f"\nindexed {len(records)} unique negatives")
    print("  skipped:")
    for k, v in skipped.most_common():
        print(f"    {v:6d}  {k}")

    by_kind = Counter(r["kind"] for r in records)
    print(f"\n  distractors : {by_kind['distractor']}")
    print(f"  background  : {by_kind['background']}")
    print("\n  by category:")
    for cat, n in Counter(r["category"] for r in records).most_common():
        print(f"    {n:5d}  {cat:24s} ({kind_of(cat)})")
    return records


def scrape_more(target_per_category=150, categories=None):
    """Top up thin negative categories using the same image search as drones."""
    import random
    import time

    import requests

    from data.build.scrape import HEADERS, MIN_SIDE, _fetch, _search
    from data.build.hashing import dhash_bytes

    existing = []
    if NEG_MANIFEST.exists():
        existing = [json.loads(l) for l in NEG_MANIFEST.open(encoding="utf-8")
                    if l.strip()]

    index = DuplicateIndex()
    counts = Counter()
    for rec in existing:
        counts[rec["category"]] += 1
        import numpy as np
        index.add(np.frombuffer(bytes.fromhex(rec["dhash"]), dtype=np.uint8),
                  rec["filename"])

    wanted = categories or list(NEGATIVE_QUERIES)
    out_root = RAW_ROOT / "negatives"
    new_records = []

    def record(rec):
        """Append immediately rather than at the end of the run.

        A scrape over 30 categories takes long enough to be interrupted, and
        buffering the manifest until the end would orphan every image already on
        disk - they would exist with nothing describing their category.
        """
        NEG_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        with NEG_MANIFEST.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        new_records.append(rec)

    for category in wanted:
        queries = queries_for(category)
        if not queries:
            print(f"{category:24s} no query defined, skipping")
            continue
        have = counts[category]
        if have >= target_per_category:
            print(f"{category:24s} have {have}, skipping")
            continue

        kind = kind_of(category) or "background"
        out_dir = out_root / category
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"{category:24s} have {have}, want {target_per_category} "
              f"({len(queries)} queries)")

        for query in queries:
            if counts[category] >= target_per_category:
                break
            need = target_per_category - counts[category]
            print(f"   \"{query}\"")
            for hit in _search(query, max(30, int(need * 2.5))):
                if counts[category] >= target_per_category:
                    break
                url = hit.get("image")
                if not url:
                    continue
                w, h = hit.get("width") or 0, hit.get("height") or 0
                if w and h and min(w, h) < MIN_SIDE:
                    continue
                raw, err = _fetch(url)
                if raw is None:
                    continue
                hsh = dhash_bytes(raw)
                if hsh is None or index.find(hsh) is not None:
                    continue

                idx = counts[category]
                fname = f"neg_{category}_{idx:04d}.jpg"
                (out_dir / fname).write_bytes(raw)
                index.add(hsh, fname)
                counts[category] += 1
                record({
                    "filename": fname,
                    "source_path": str((out_dir / fname).as_posix()),
                    "category": category,
                    "kind": kind,
                    "distractor_class": distractor_class(category),
                    "width": w,
                    "height": h,
                    "dhash": hsh.tobytes().hex(),
                })
            time.sleep(random.uniform(1.0, 2.0))
        print(f"      total {counts[category]}")

    print(f"\nadded {len(new_records)} new negatives "
          f"({len(existing) + len(new_records)} total)")
    return new_records


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--harvest", action="store_true",
                   help="index unique negatives already on disk")
    p.add_argument("--scrape", action="store_true",
                   help="download more negatives for thin categories")
    p.add_argument("--target", type=int, default=150)
    p.add_argument("--sources", nargs="*", default=["Negative"])
    p.add_argument("--categories", nargs="*", default=None)
    a = p.parse_args()

    if not (a.harvest or a.scrape):
        p.error("pass --harvest and/or --scrape")
    if a.harvest:
        harvest(source_dirs=a.sources)
    if a.scrape:
        scrape_more(target_per_category=a.target, categories=a.categories)


if __name__ == "__main__":
    main()
