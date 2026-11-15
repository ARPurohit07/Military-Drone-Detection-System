"""Legacy stage-2 experiment: what kind of drone is in this crop?

Run with:
    python -m tools.crop_classifier --scheme airframe
    python -m tools.crop_classifier --scheme country
    python -m tools.crop_classifier --scheme role

Why a second model at all
-------------------------
This experiment predates the v3 two-Faster-R-CNN pipeline. Its metrics describe
the earlier dataset and must not be quoted as current benchmark results. The
current held-out numbers are in reports/country_v3_diagnosis.txt.

The reason is structural. Faster R-CNN classifies from a 7x7 ROI-pooled feature
map through a small MLP - about 49 spatial samples to separate a Wing Loong from a
Bayraktar, when the distinguishing evidence is the shape of a sensor ball, a V-tail
versus twin booms, or where the engine sits. That is enough for "car versus
person" and far too little here.

So: crop the box, resize to a proper classification resolution, and train a real
backbone on it. The detector keeps doing what it is good at.

This also serves as a cheap probe for the airframe taxonomy. Retraining the whole
detector under --scheme airframe costs ~8 hours; training this on crops costs
minutes, and it answers the same question - is airframe visually learnable, and
does deriving country from a predicted airframe beat predicting country directly?

Country is evaluated three ways so the comparison is honest:
  - direct       : trained to predict country
  - via airframe : trained to predict airframe, country looked up in the registry
  - via role     : trained to predict role (sanity check that role is easier)
"""

import argparse
import collections
import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset

from data.airframes import BY_KEY, class_for

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASET = Path("drone_dataset_v2")
# Crops are small and there are only ~8k of them, so a modest backbone at a real
# classification resolution is affordable where a detector backbone is not.
BACKBONE = "convnext_tiny"
CROP_SIZE = 224
# Crops are expanded slightly beyond the box: the auto-labelled boxes already cut
# thin wings and props (see reports/sam_refinement_rejected.txt for how badly a
# mask-tightened box does), and context helps a classifier judge scale.
BOX_PAD = 0.15


def read_annotation(xml_path):
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    W = int(float(size.find("width").text))
    H = int(float(size.find("height").text))
    src = root.find("source")
    meta = {}
    if src is not None:
        for child in src:
            if child.text:
                meta[child.tag] = child.text
    objs = []
    for o in root.findall("object"):
        name_el = o.find("name")
        name = (name_el.text or "").strip() if name_el is not None else ""
        if not name:
            continue
        sc = o.find("score")
        score = float(sc.text) if sc is not None and sc.text else 1.0
        bb = o.find("bndbox")
        box = [float(bb.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")]
        objs.append((box, score))
    return W, H, meta, objs


class CropDataset(Dataset):
    """One sample per annotated box, labelled under the chosen scheme."""

    def __init__(self, split, scheme, label_map, train=False,
                 min_score=0.25, max_area_frac=0.75):
        self.train = train
        self.samples = []
        img_dir = DATASET / "images" / split
        ann_dir = DATASET / "annotations" / split
        for xml in sorted(ann_dir.glob("*.xml")):
            img = None
            for ext in (".jpg", ".jpeg", ".png"):
                cand = img_dir / (xml.stem + ext)
                if cand.exists():
                    img = cand
                    break
            if img is None:
                continue
            W, H, meta, objs = read_annotation(xml)
            airframe = meta.get("airframe")
            if not airframe or airframe not in BY_KEY:
                continue            # negatives and anything unmapped
            cls = class_for(airframe, scheme)
            if cls not in label_map:
                continue
            for box, score in objs:
                if score < min_score:
                    continue
                area = ((box[2]-box[0]) * (box[3]-box[1])) / max(1.0, W*H)
                if area > max_area_frac:
                    continue
                if box[2]-box[0] < 8 or box[3]-box[1] < 8:
                    continue
                self.samples.append({
                    "img": img, "box": box, "wh": (W, H),
                    "label": label_map[cls], "airframe": airframe,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s["img"]).convert("RGB")
        x1, y1, x2, y2 = s["box"]
        w, h = x2 - x1, y2 - y1
        pad = BOX_PAD
        if self.train:
            pad += random.uniform(-0.05, 0.10)
        x1 = max(0, x1 - w * pad)
        y1 = max(0, y1 - h * pad)
        x2 = min(s["wh"][0], x2 + w * pad)
        y2 = min(s["wh"][1], y2 + h * pad)
        crop = img.crop((x1, y1, x2, y2)).resize((CROP_SIZE, CROP_SIZE),
                                                 Image.BILINEAR)
        arr = np.asarray(crop, dtype=np.float32) / 255.0
        if self.train:
            if random.random() < 0.5:
                arr = arr[:, ::-1].copy()
            arr = np.clip(arr * random.uniform(0.75, 1.25), 0, 1)
            if random.random() < 0.3:                 # mimic the eval degradations
                noise = np.random.normal(0, 0.03, arr.shape).astype(np.float32)
                arr = np.clip(arr + noise, 0, 1)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        # Cast explicitly: numpy promotes to float64 the moment any float64
        # intermediate creeps in (np.random.normal defaults to it), and a double
        # tensor fails inside autocast with "Input type (double) and bias type
        # (struct c10::Half) should be the same".
        arr = np.ascontiguousarray(arr.transpose(2, 0, 1), dtype=np.float32)
        return torch.from_numpy(arr), s["label"]


def build_label_map(scheme, min_count):
    """Label map over classes with enough boxes in train to be learnable."""
    counts = collections.Counter()
    for xml in sorted((DATASET / "annotations" / "train").glob("*.xml")):
        _, _, meta, objs = read_annotation(xml)
        af = meta.get("airframe")
        if af and af in BY_KEY and objs:
            counts[class_for(af, scheme)] += len(objs)
    kept = sorted(c for c, n in counts.items() if n >= min_count)
    dropped = sorted(c for c, n in counts.items() if n < min_count)
    return {c: i for i, c in enumerate(kept)}, dropped, counts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scheme", default="airframe",
                    choices=["airframe", "country", "role"])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-count", type=int, default=20)
    ap.add_argument("--backbone", default=BACKBONE)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    label_map, dropped, counts = build_label_map(args.scheme, args.min_count)
    names = {v: k for k, v in label_map.items()}
    print(f"scheme={args.scheme}  classes={len(label_map)}  device={dev}")
    if dropped:
        print(f"dropped {len(dropped)} classes under {args.min_count} boxes: "
              f"{dropped[:8]}{'...' if len(dropped) > 8 else ''}")

    tr = CropDataset("train", args.scheme, label_map, train=True)
    va = CropDataset("val", args.scheme, label_map, train=False)
    print(f"train crops {len(tr)}   val crops {len(va)}")
    if not len(tr) or not len(va):
        raise SystemExit("no crops - was the dataset assembled?")

    # Inverse-frequency sampling: airframe classes hold ~57 train crops each and
    # the spread across classes is wide enough that the loss would otherwise be
    # dominated by the largest.
    lab = [s["label"] for s in tr.samples]
    freq = collections.Counter(lab)
    weights = [1.0 / freq[l] ** 0.5 for l in lab]
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights),
                                                     replacement=True)

    dl_tr = DataLoader(tr, batch_size=args.batch_size, sampler=sampler,
                       num_workers=0, drop_last=True)
    dl_va = DataLoader(va, batch_size=args.batch_size, shuffle=False,
                       num_workers=0)

    model = timm.create_model(args.backbone, pretrained=True,
                              num_classes=len(label_map)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    scaler = torch.amp.GradScaler("cuda") if dev == "cuda" else None
    # Label smoothing: several of these airframes genuinely look alike, so forcing
    # a hard one-hot target over-penalises a defensible confusion.
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    best = 0.0
    best_state = None
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        for x, y in dl_tr:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.float16,
                                enabled=scaler is not None):
                loss = crit(model(x), y)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            tot += loss.item()
        sched.step()

        model.eval()
        top1 = top3 = n = 0
        per_cls = collections.Counter()
        per_cls_ok = collections.Counter()
        preds_all, labs_all = [], []
        with torch.no_grad():
            for x, y in dl_va:
                x = x.to(dev)
                with torch.autocast("cuda", dtype=torch.float16,
                                    enabled=scaler is not None):
                    logits = model(x)
                top = logits.float().topk(min(3, logits.shape[1]), dim=1).indices.cpu()
                for i, t in enumerate(y):
                    n += 1
                    per_cls[int(t)] += 1
                    if int(top[i, 0]) == int(t):
                        top1 += 1
                        per_cls_ok[int(t)] += 1
                    if int(t) in [int(v) for v in top[i]]:
                        top3 += 1
                    preds_all.append(int(top[i, 0]))
                    labs_all.append(int(t))
        acc1, acc3 = 100 * top1 / n, 100 * top3 / n
        print(f"epoch {ep:2d}  loss {tot/max(1,len(dl_tr)):.4f}  "
              f"val top1 {acc1:5.2f}%  top3 {acc3:5.2f}%", flush=True)
        if acc1 > best:
            best = acc1
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            best_report = (per_cls, per_cls_ok, preds_all, labs_all)

    print(f"\nbest val top-1: {best:.2f}%  (chance {100/len(label_map):.2f}%)")

    # If the scheme is finer than country, roll the prediction up through the
    # registry: a correct airframe implies a correct country by construction.
    if args.scheme in ("airframe", "role"):
        per_cls, per_cls_ok, preds_all, labs_all = best_report
        if args.scheme == "airframe":
            to_country = {i: BY_KEY[names[i]].country for i in names}
            ok = sum(1 for p, t in zip(preds_all, labs_all)
                     if to_country[p] == to_country[t])
            n_countries = len(set(to_country.values()))
            print(f"\nCOUNTRY derived from predicted airframe: "
                  f"{100*ok/len(labs_all):.2f}%  "
                  f"over {n_countries} countries "
                  f"(direct-country detector scored 31.3%)")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": best_state, "label_map": label_map,
                    "scheme": args.scheme, "backbone": args.backbone,
                    "crop_size": CROP_SIZE, "box_pad": BOX_PAD,
                    "val_top1": best}, args.out)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
