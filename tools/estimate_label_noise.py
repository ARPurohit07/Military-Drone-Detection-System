"""How many scraped crops actually contain a military drone?

Run with:  python -m tools.estimate_label_noise

The legacy airframe crop classifier reached 2.1% top-1 against a 0.83% chance baseline,
with training loss barely moving off ln(120). Inspecting the crops showed why: a
sample of 24 contained a warship, a laptop, a car and a corporate sign, each
carrying a specific airframe label. Image search for an obscure airframe
("Shahpar-II UAV Pakistan") returns whatever it has, the auto-labeller boxes the
most drone-like thing present, and the registry stamps the airframe name on it.

The auto-labeller only ever asked "is there a drone here"; it never asked "is this
the airframe the query claimed". So this scores every crop with CLIP against a set
of competing descriptions and reports what fraction look like a military drone at
all - turning an impression from 24 tiles into a number over thousands.

This measures presence of a drone, NOT airframe identity. A crop that scores well
here can still be the wrong drone, so the figure it produces is an upper bound on
label quality, not an estimate of it.
"""

import argparse
import collections
import random

import numpy as np
import torch
from PIL import Image, ImageFile
from transformers import CLIPModel, CLIPProcessor

from tools.crop_classifier import CropDataset, build_label_map

ImageFile.LOAD_TRUNCATED_IMAGES = True

MODEL = "openai/clip-vit-base-patch32"

# Deliberately includes the wrong-content categories actually seen in the crops,
# so a laptop or a warship has somewhere to go other than "drone".
PROMPTS = [
    "a photo of a military drone or unmanned aerial vehicle",
    "a photo of a model airplane or remote control aircraft",
    "a photo of a manned airplane or jet fighter",
    "a photo of a helicopter",
    "a photo of a missile or rocket",
    "a photo of a ship or boat",
    "a photo of a car or ground vehicle",
    "a photo of a person or people",
    "a photo of a computer, screen or office equipment",
    "a photo of a building, sign or logo",
    "a diagram, chart or infographic",
    "a photo of sky, clouds or landscape with nothing in it",
]
DRONE_IDX = 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=int, default=1200)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = CLIPProcessor.from_pretrained(MODEL)
    model = CLIPModel.from_pretrained(MODEL).to(dev).eval()

    label_map, _, _ = build_label_map("airframe", 1)
    names = {v: k for k, v in label_map.items()}
    ds = CropDataset("train", "airframe", label_map, train=False)
    idx = list(range(len(ds)))
    random.seed(args.seed)
    random.shuffle(idx)
    idx = idx[:args.sample]
    print(f"scoring {len(idx)} crops of {len(ds)} with CLIP on {dev}\n",
          flush=True)

    wins = collections.Counter()
    per_airframe = collections.defaultdict(lambda: [0, 0])
    drone_scores = []

    for start in range(0, len(idx), args.batch_size):
        chunk = idx[start:start + args.batch_size]
        images, labels = [], []
        for i in chunk:
            s = ds.samples[i]
            try:
                im = Image.open(s["img"]).convert("RGB")
            except Exception:                                  # noqa: BLE001
                continue
            x1, y1, x2, y2 = s["box"]
            images.append(im.crop((x1, y1, x2, y2)))
            labels.append(names[s["label"]])
        if not images:
            continue
        # Full forward rather than get_*_features: in transformers 5.x those
        # return a BaseModelOutputWithPooling whose pooler_output is taken BEFORE
        # the projection head, so normalising it would compare the wrong vectors.
        # logits_per_image is the canonical, projection-correct similarity and is
        # stable across versions.
        inp = proc(text=PROMPTS, images=images, return_tensors="pt",
                   padding=True).to(dev)
        with torch.no_grad():
            sims = model(**inp).logits_per_image.softmax(dim=-1).cpu().numpy()
        for row, lab in zip(sims, labels):
            w = int(row.argmax())
            wins[PROMPTS[w]] += 1
            drone_scores.append(float(row[DRONE_IDX]))
            per_airframe[lab][1] += 1
            if w == DRONE_IDX:
                per_airframe[lab][0] += 1
        if (start + len(chunk)) % 320 < args.batch_size:
            print(f"   {start + len(chunk)}/{len(idx)}", flush=True)

    total = sum(wins.values())
    drone = wins[PROMPTS[DRONE_IDX]]
    print("\n" + "=" * 66)
    print("WHAT DOES CLIP THINK THESE CROPS ARE?")
    print("=" * 66)
    for p, n in wins.most_common():
        print(f"  {100*n/total:5.1f}%  {n:5d}  {p}")
    print(f"\n  looks like a drone : {100*drone/total:.1f}%")
    print(f"  looks like something else: {100*(total-drone)/total:.1f}%")
    print(f"  mean P(drone): {np.mean(drone_scores):.3f}")

    ranked = sorted(((v[0]/max(v[1], 1), v[1], k)
                     for k, v in per_airframe.items() if v[1] >= 4))
    print("\n  worst airframes by fraction that look like a drone:")
    for frac, n, k in ranked[:15]:
        print(f"    {100*frac:5.1f}%  (n={n:2d})  {k}")
    print("\n  best:")
    for frac, n, k in ranked[-8:]:
        print(f"    {100*frac:5.1f}%  (n={n:2d})  {k}")

    print("\nNOTE: this measures whether a drone is present, not whether it is the")
    print("airframe the label claims. Real airframe-label accuracy is lower.")


if __name__ == "__main__":
    main()
