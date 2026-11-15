"""Does SAM-refining a box actually make it tighter, or just different?

Run with:  python -m tools.validate_sam_refine

The auto-labeller's boxes agree with human annotations at median IoU ~0.66, which
caps what any detector trained on them can learn - the model cannot be more
precise than its labels. SAM is the obvious fix: prompt it with the loose box,
take the tight bounding box of the returned mask.

"Obvious" is not "verified", and this is cheap to check the right way. The legacy
dataset has human boxes, so: run the auto-labeller on those images, refine with
SAM, and compare BOTH against the human box. If refinement helps, IoU goes up. It
could equally go down - SAM might segment the sky around a silhouette, or latch
onto a launch rail or a person standing beside the aircraft - and that would be
worth knowing before rewriting 8,271 annotations.
"""
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile
from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor,
                          SamModel, SamProcessor)

ImageFile.LOAD_TRUNCATED_IMAGES = True

DINO = "IDEA-Research/grounding-dino-tiny"
SAM = "facebook/sam-vit-base"
PROMPT = "a drone. an unmanned aerial vehicle."
LEGACY = Path("drone_dataset_3_test")
N_SAMPLE = 60
SEED = 0

dev = "cuda" if torch.cuda.is_available() else "cpu"


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)
    return inter / ua if ua > 0 else 0.0


def load_samples():
    out = []
    for split in ("train", "val"):
        for xml in sorted((LEGACY / "annotations" / split).glob("*.xml")):
            for ext in (".jpg", ".png", ".jpeg"):
                img = LEGACY / "images" / split / (xml.stem + ext)
                if img.exists():
                    break
            else:
                continue
            root = ET.parse(xml).getroot()
            size = root.find("size")
            W = float(size.find("width").text)
            H = float(size.find("height").text)
            gt = [[float(o.find("bndbox").find(k).text)
                   for k in ("xmin", "ymin", "xmax", "ymax")]
                  for o in root.findall("object")]
            if not gt or W <= 0 or H <= 0:
                continue
            # Skip the known-bad near-full-frame human boxes.
            if max(((g[2]-g[0])*(g[3]-g[1]))/(W*H) for g in gt) > 0.75:
                continue
            out.append((img, gt))
    return out


def main():
    samples = load_samples()
    random.seed(SEED)
    random.shuffle(samples)
    samples = samples[:N_SAMPLE]
    print(f"{len(samples)} images with clean human boxes, device={dev}\n",
          flush=True)

    dproc = AutoProcessor.from_pretrained(DINO)
    dmodel = AutoModelForZeroShotObjectDetection.from_pretrained(DINO).to(dev).eval()
    sproc = SamProcessor.from_pretrained(SAM)
    smodel = SamModel.from_pretrained(SAM).to(dev).eval()
    print("models loaded\n", flush=True)

    raw_ious, ref_ious = [], []
    grew, shrank, failed = 0, 0, 0

    for n, (path, gt) in enumerate(samples):
        img = Image.open(path).convert("RGB")

        inp = dproc(images=img, text=PROMPT, return_tensors="pt",
                    size={"shortest_edge": 480, "longest_edge": 800}).to(dev)
        with torch.no_grad():
            out = dmodel(**inp)
        res = dproc.post_process_grounded_object_detection(
            out, inp.input_ids, threshold=0.25, text_threshold=0.25,
            target_sizes=[img.size[::-1]])[0]
        if len(res["boxes"]) == 0:
            continue
        k = int(res["scores"].argmax())
        raw = [float(v) for v in res["boxes"][k]]

        # Prompt SAM with the loose box and take the mask's tight bounds.
        sin = sproc(img, input_boxes=[[raw]], return_tensors="pt").to(dev)
        with torch.no_grad():
            sout = smodel(**sin, multimask_output=False)
        masks = sproc.image_processor.post_process_masks(
            sout.pred_masks.cpu(), sin["original_sizes"].cpu(),
            sin["reshaped_input_sizes"].cpu())[0]
        mask = masks[0][0].numpy()

        ys, xs = np.where(mask)
        if len(xs) == 0:
            failed += 1
            refined = raw
        else:
            refined = [float(xs.min()), float(ys.min()),
                       float(xs.max()), float(ys.max())]
            ra = (raw[2]-raw[0]) * (raw[3]-raw[1])
            fa = (refined[2]-refined[0]) * (refined[3]-refined[1])
            if fa > ra * 1.05:
                grew += 1
            elif fa < ra * 0.95:
                shrank += 1

        raw_ious.append(max(iou(raw, g) for g in gt))
        ref_ious.append(max(iou(refined, g) for g in gt))
        if (n + 1) % 20 == 0:
            print(f"   {n+1}/{len(samples)}", flush=True)

    a, b = np.array(raw_ious), np.array(ref_ious)
    print("\n" + "=" * 64)
    print("SAM BOX REFINEMENT vs HUMAN BOXES")
    print("=" * 64)
    print(f"  compared on {len(a)} boxes")
    print(f"{'':22s}{'raw DINO':>12s}{'SAM-refined':>14s}{'change':>10s}")
    print(f"  {'mean IoU':20s}{a.mean():12.3f}{b.mean():14.3f}"
          f"{b.mean()-a.mean():+10.3f}")
    print(f"  {'median IoU':20s}{np.median(a):12.3f}{np.median(b):14.3f}"
          f"{np.median(b)-np.median(a):+10.3f}")
    for t in (0.5, 0.75, 0.9):
        pa, pb = 100*(a >= t).mean(), 100*(b >= t).mean()
        print(f"  {'IoU >= ' + str(t):20s}{pa:11.1f}%{pb:13.1f}%{pb-pa:+9.1f}%")
    print(f"\n  mask grew the box : {grew}")
    print(f"  mask shrank it    : {shrank}")
    print(f"  mask empty (kept raw): {failed}")
    better = int((b > a + 0.02).sum())
    worse = int((b < a - 0.02).sum())
    print(f"\n  refinement helped on {better} boxes, hurt on {worse}, "
          f"neutral on {len(a)-better-worse}")
    verdict = ("WORTH APPLYING" if b.mean() > a.mean() + 0.01
               else "NOT WORTH APPLYING - no meaningful gain")
    print(f"\n  verdict: {verdict}")


if __name__ == "__main__":
    main()
