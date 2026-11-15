"""Verify assemble's conflict quarantine on synthetic collisions.

Run with:  python -m tests.test_quarantine


Real scraped data has produced zero conflicts so far, so the mechanism that
exists specifically to prevent the original dataset's defect - the same picture
filed under two countries - has not actually been exercised. This builds those
collisions deliberately:

  1. identical image content under two different airframes (different countries)
  2. a re-encoded (not byte-identical) copy, to prove perceptual rather than
     exact matching
  3. the same picture present as both a drone and a hard negative
  4. a clean control that must survive

Runs against a temporary RAW_ROOT/OUT_ROOT so the live build is untouched.
"""
import json, shutil, sys, tempfile
from pathlib import Path
from xml.dom import minidom
import xml.etree.ElementTree as ET

import cv2
import numpy as np

import data.build.assemble as A
from data.build.hashing import dhash_file

tmp = Path(tempfile.mkdtemp(prefix="qtest_"))
RAW = tmp / "raw"
OUT = tmp / "out"
(RAW / "images").mkdir(parents=True)
(RAW / "annotations").mkdir(parents=True)
(RAW / "negatives").mkdir(parents=True)

rng = np.random.default_rng(7)


def make_image(path, seed):
    """A textured image - flat noise can hash-collide, real photos do not."""
    r = np.random.default_rng(seed)
    base = (r.random((240, 320, 3)) * 255).astype(np.uint8)
    base = cv2.GaussianBlur(base, (9, 9), 0)
    cv2.rectangle(base, (60, 50), (200, 150), (240, 240, 240), -1)
    cv2.circle(base, (160, 100), 30, (20, 20, 20), -1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), base)


def write_ann(airframe, stem, w=320, h=240):
    ann = ET.Element("annotation")
    ET.SubElement(ann, "filename").text = stem + ".jpg"
    size = ET.SubElement(ann, "size")
    ET.SubElement(size, "width").text = str(w)
    ET.SubElement(size, "height").text = str(h)
    obj = ET.SubElement(ann, "object")
    ET.SubElement(obj, "name").text = airframe
    ET.SubElement(obj, "difficult").text = "0"
    ET.SubElement(obj, "score").text = "0.9"
    bb = ET.SubElement(obj, "bndbox")
    for k, v in zip(("xmin", "ymin", "xmax", "ymax"), (60, 50, 200, 150)):
        ET.SubElement(bb, k).text = str(v)
    p = RAW / "annotations" / airframe / (stem + ".xml")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(minidom.parseString(ET.tostring(ann)).toprettyxml(indent="  "),
                 encoding="utf-8")


drones, negatives = [], []


def add_drone(airframe, country, role, stem, src_seed=None, copy_of=None,
              reencode=False):
    img = RAW / "images" / airframe / (stem + ".jpg")
    if copy_of is not None:
        img.parent.mkdir(parents=True, exist_ok=True)
        if reencode:
            # Re-encode at lower quality: different bytes, same picture. Exact
            # md5 matching would miss this; a perceptual hash must not.
            a = cv2.imread(str(copy_of))
            cv2.imwrite(str(img), a, [int(cv2.IMWRITE_JPEG_QUALITY), 55])
        else:
            shutil.copy2(copy_of, img)
    else:
        make_image(img, src_seed)
    write_ann(airframe, stem)
    h = dhash_file(img)
    drones.append({
        "filename": stem + ".jpg", "path": str(img.as_posix()),
        "airframe": airframe, "airframe_name": airframe, "country": country,
        "role": role, "query": "synthetic", "image_url": "x",
        "width": 320, "height": 240, "dhash": h.tobytes().hex(),
    })
    return img


def add_negative(category, stem, copy_of=None, src_seed=None):
    img = RAW / "negatives" / category / (stem + ".jpg")
    if copy_of is not None:
        img.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(copy_of, img)
    else:
        make_image(img, src_seed)
    h = dhash_file(img)
    negatives.append({
        "filename": stem + ".jpg", "source_path": str(img.as_posix()),
        "category": category, "kind": "distractor",
        "distractor_class": "bird", "width": 320, "height": 240,
        "dhash": h.tobytes().hex(),
    })


# 1. identical content, two countries  -> must quarantine
a = add_drone("bayraktar_tb2", "Turkey", "male", "tb2_dup", src_seed=1)
add_drone("shahed_136", "Iran", "loitering_munition", "shahed_dup", copy_of=a)

# 2. re-encoded copy, two countries    -> must quarantine (perceptual, not md5)
b = add_drone("mq9_reaper", "USA", "male", "reaper_dup", src_seed=2)
add_drone("wing_loong_2", "China", "male", "wl2_dup", copy_of=b, reencode=True)

# 3. same picture as drone AND negative -> must quarantine
c = add_drone("iai_harop", "Israel", "loitering_munition", "harop_dup",
              src_seed=3)
add_negative("birds_in_sky", "neg_collide", copy_of=c)

# 4. clean controls                     -> must survive
for i, (af, ctry) in enumerate([("bayraktar_tb2", "Turkey"),
                                ("shahed_136", "Iran"),
                                ("mq9_reaper", "USA")]):
    add_drone(af, ctry, "male", f"clean_{af}_{i}", src_seed=100 + i)
add_negative("birds_in_sky", "neg_clean", src_seed=200)

(RAW / "manifest.jsonl").write_text(
    "\n".join(json.dumps(d) for d in drones) + "\n", encoding="utf-8")
(RAW / "negatives.jsonl").write_text(
    "\n".join(json.dumps(n) for n in negatives) + "\n", encoding="utf-8")

A.RAW_ROOT = RAW
A.MANIFEST = RAW / "manifest.jsonl"
A.ANN_ROOT = RAW / "annotations"
A.OUT_ROOT = OUT
A.REPORT = tmp / "report.txt"

print(f"synthetic corpus: {len(drones)} drone entries, "
      f"{len(negatives)} negatives")
print("expect 3 conflict groups quarantined, 4 clean pictures kept\n")

A.assemble(scheme="country", val_fraction=0.25, min_per_class=1, seed=1)

print("\n================ CHECKS ================")
conflicts = OUT / "quarantine" / "conflicts.jsonl"
rows = [json.loads(l) for l in conflicts.open(encoding="utf-8")] \
    if conflicts.exists() else []
groups = {r["group"] for r in rows}
names = {r["filename"] for r in rows}

kept = {p.name for p in (OUT / "images").rglob("*.jpg")}

checks = [
    ("3 conflict groups quarantined", len(groups) == 3, f"got {len(groups)}"),
    ("identical-content pair quarantined",
     {"tb2_dup.jpg", "shahed_dup.jpg"} <= names, sorted(names)),
    ("RE-ENCODED pair quarantined (perceptual, not md5)",
     {"reaper_dup.jpg", "wl2_dup.jpg"} <= names, sorted(names)),
    ("drone/negative collision quarantined",
     {"harop_dup.jpg", "neg_collide.jpg"} <= names, sorted(names)),
    ("no quarantined file leaked into the dataset",
     not (names & kept), sorted(names & kept)),
    ("clean controls survived",
     sum(1 for k in kept if k.startswith("clean_")) == 3,
     sorted(k for k in kept if k.startswith("clean_"))),
    ("clean negative survived", "neg_clean.jpg" in kept, sorted(kept)),
]

ok = True
for label, passed, detail in checks:
    print(f"  {'PASS' if passed else 'FAIL'}  {label}")
    if not passed:
        print(f"        -> {detail}")
        ok = False

shutil.rmtree(tmp, ignore_errors=True)
print("\n" + ("ALL QUARANTINE CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
sys.exit(0 if ok else 1)
