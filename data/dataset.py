"""Dataset loader for the rebuilt corpus.

The loader got noticeably simpler, because the *dataset* now carries information
that the loader used to have to invent.

Negatives are ordinary samples. Every image in the rebuilt dataset has an
annotation file, and a negative's annotation simply contains zero objects. The
old loader had to special-case a `images/negatives/` directory that sat outside
the train/val split, so all 200 negatives were loaded into both splits at once -
the model was trained on the very images its false-positive rate was measured
on. Splitting them is now the builder's job, done once, and recorded on disk.

Provenance travels with the sample. Each annotation records the airframe,
country, role and the search query behind it, so `sample["meta"]` can answer
"which airframe is this, and how did it get labelled" without a lookup table.
That is what makes per-airframe or per-role evaluation possible against the same
files.

`difficult` is meaningful again. Auto-labelled boxes whose detector confidence
fell below the confident threshold carry VOC's `difficult` flag. Training skips
them by default (`skip_difficult`) so it only sees trustworthy geometry, while
evaluation can opt to include them.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

from config import CLASS_MAP

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


class DroneDataset(Dataset):
    """VOC-style detection dataset.

    Args:
        root: dataset directory holding `images/<split>` and `annotations/<split>`.
        split: "train" or "val".
        transforms: callable taking `(image, target)`.
        skip_difficult: drop boxes flagged difficult (low-confidence auto-labels).
        include_negatives: keep images with zero objects. Negatives are part of
            the split on disk, so this only filters what is already there - it
            cannot pull in images from the other split the way the old flag did.
        keep_empty_ratio: cap negatives at this fraction of the split. None keeps
            all of them. Useful because too many background images relative to
            drones shifts the RPN's positive/negative balance and suppresses
            recall.
    """

    def __init__(self, root, split, transforms=None, skip_difficult=True,
                 include_negatives=True, keep_empty_ratio=None,
                 min_score=None, max_area_frac=None, binary=False,
                 label_map=None):
        self.root = Path(root)
        self.split = split
        self.transforms = transforms
        self.skip_difficult = skip_difficult
        # Binary mode collapses every foreground class to a single "drone" label.
        #
        # Detection and attribution are separate problems with very different
        # accuracy (binary mAP@0.5 0.9503 versus 33.6% country top-1 given a
        # localized object), so they are better served by separate models.
        #
        # This collapses the labels at load time rather than rebuilding the
        # dataset: the annotations already carry airframe and country in their
        # <source> block, so the same files on disk serve the detector and the
        # stage-2 detector without duplicating the source images.
        self.binary = binary
        self.label_map = dict(label_map or CLASS_MAP)
        # `difficult` conflates two unrelated things: the detector was unsure
        # (score below the confident threshold) and the box covers most of the
        # frame. Those deserve different treatment, so they can be filtered
        # separately here.
        #
        # A low-confidence box is usually the right object located imprecisely,
        # and dropping it often drops the whole image - when every box in an image
        # is flagged, the image has no objects left and is discarded entirely,
        # losing the class signal too. That cost 19-32% of images per class, worst
        # on the thinnest classes.
        #
        # A near-full-frame box is the defect that made the old dataset teach
        # "drone means the whole picture", and is worth excluding regardless of
        # confidence.
        #
        # Setting `min_score` switches to score-based filtering and ignores the
        # flag; `max_area_frac` always applies when set.
        self.min_score = min_score
        self.max_area_frac = max_area_frac

        self.img_dir = self.root / "images" / split
        self.ann_dir = self.root / "annotations" / split
        if not self.ann_dir.exists():
            raise FileNotFoundError(
                f"no annotations at {self.ann_dir} - build the dataset first "
                f"with `python -m data.build.pipeline`"
            )

        positives, negatives = [], []
        self.n_dropped_unlabelled = 0
        for xml_path in sorted(self.ann_dir.glob("*.xml")):
            img_path = self._find_image(xml_path.stem)
            if img_path is None:
                continue
            n_objects, n_named, meta = self._peek(xml_path)
            sample = {
                "stem": xml_path.stem,
                "img_path": img_path,
                "xml_path": xml_path,
                "is_negative": n_objects == 0,
                "meta": meta,
            }
            if n_objects == 0:
                # Distinguish a real hard negative from a drone image that merely
                # lost all of its boxes to the `difficult` filter. Treating the
                # latter as background would teach the model that this airframe
                # is NOT a drone - the exact opposite of the intended signal - so
                # it is dropped instead.
                provenance = meta.get("provenance")
                source_classes = {
                    name.strip().lower()
                    for name in meta.get("source_classes", "").split(",")
                    if name.strip()
                }
                declared_negative = (
                    meta.get("kind") == "negative"
                    or provenance in {"negative", "deployment_mined"}
                    # UAV_SMID distractor images carry their original source
                    # class names; a difficult/oversized drone still says
                    # source_classes=drone and must be dropped, not converted
                    # into a false negative merely because it has a category.
                    or (meta.get("source_dataset") == "UAV_SMID"
                        and "drone" not in source_classes)
                )
                if not declared_negative:
                    self.n_dropped_unlabelled += 1
                    continue
                negatives.append(sample)
            else:
                positives.append(sample)

        if not include_negatives:
            negatives = []
        elif keep_empty_ratio is not None and positives:
            cap = int(keep_empty_ratio * len(positives))
            negatives = negatives[:cap]

        self.samples = positives + negatives
        self.n_positive = len(positives)
        self.n_negative = len(negatives)

    def _find_image(self, stem):
        for ext in IMAGE_EXTENSIONS:
            candidate = self.img_dir / (stem + ext)
            if candidate.exists():
                return candidate
        return None

    def _peek(self, xml_path):
        """Count objects and read provenance, without loading pixels.

        Returns `(n_trainable, n_named, meta)`. The two counts differ when boxes
        were dropped by the `difficult` filter, which is what tells a genuine
        negative apart from a drone image whose labels were all low-confidence.
        """
        root = ET.parse(xml_path).getroot()
        size = root.find("size")
        try:
            width = int(float(size.find("width").text))
            height = int(float(size.find("height").text))
        except (AttributeError, TypeError, ValueError):
            width = height = None

        n = 0
        n_named = 0
        for obj in root.findall("object"):
            name_el = obj.find("name")
            if name_el is None or not (name_el.text or "").strip():
                continue    # a negative's placeholder object, if any
            n_named += 1
            if self._rejects(obj, width, height):
                continue
            n += 1

        meta = {}
        source = root.find("source")
        if source is not None:
            for child in source:
                if child.text:
                    meta[child.tag] = child.text
        return n, n_named, meta

    @staticmethod
    def _is_difficult(obj):
        el = obj.find("difficult")
        if el is None or not el.text:
            return False
        return int(el.text) == 1

    @staticmethod
    def _box_score(obj):
        el = obj.find("score")
        if el is None or not el.text:
            return 1.0
        try:
            return float(el.text)
        except ValueError:
            return 1.0

    def _rejects(self, obj, width=None, height=None):
        """Whether this object should be excluded from training.

        One predicate, used by both the index pass and the pixel pass, so the
        count of trainable boxes cannot drift from the boxes actually returned.
        """
        if self.min_score is not None:
            if self._box_score(obj) < self.min_score:
                return True
        elif self.skip_difficult and self._is_difficult(obj):
            return True

        if self.max_area_frac is not None and width and height:
            bb = obj.find("bndbox")
            if bb is not None:
                try:
                    x1 = float(bb.find("xmin").text)
                    y1 = float(bb.find("ymin").text)
                    x2 = float(bb.find("xmax").text)
                    y2 = float(bb.find("ymax").text)
                except (AttributeError, TypeError, ValueError):
                    return False
                frac = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1)) / float(width * height)
                if frac > self.max_area_frac:
                    return True
        return False

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Read annotation dimensions before pixels so very large JPEG/PNG files
        # can use OpenCV's reduced-resolution decoder.  Resizing after a normal
        # imread is too late: an 8K RGB photograph first allocates ~130 MB and
        # can exhaust Windows commit before ResizeLongestSide ever runs.
        root = ET.parse(sample["xml_path"]).getroot()
        size = root.find("size")
        try:
            source_width = int(float(size.find("width").text))
            source_height = int(float(size.find("height").text))
        except (AttributeError, TypeError, ValueError):
            source_width = source_height = 0

        longest = max(source_width, source_height)
        if longest > 6400:
            read_flag = cv2.IMREAD_REDUCED_COLOR_8
        elif longest > 3200:
            read_flag = cv2.IMREAD_REDUCED_COLOR_4
        elif longest > 1600:
            read_flag = cv2.IMREAD_REDUCED_COLOR_2
        else:
            read_flag = cv2.IMREAD_COLOR

        img = cv2.imread(str(sample["img_path"]), read_flag)
        if img is None:
            raise RuntimeError(f"unreadable image: {sample['img_path']}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width = img.shape[:2]
        annotation_width = source_width or width
        annotation_height = source_height or height
        scale_x = width / float(annotation_width)
        scale_y = height / float(annotation_height)

        boxes, labels = [], []
        for obj in root.findall("object"):
            name_el = obj.find("name")
            name = (name_el.text or "").strip() if name_el is not None else ""
            if not name:
                continue
            if self._rejects(obj, annotation_width, annotation_height):
                continue

            bb = obj.find("bndbox")
            xmin = float(bb.find("xmin").text) * scale_x
            ymin = float(bb.find("ymin").text) * scale_y
            xmax = float(bb.find("xmax").text) * scale_x
            ymax = float(bb.find("ymax").text) * scale_y

            # Clamp to the image and drop degenerate boxes. A zero-area or
            # inverted box makes torchvision's loss produce NaN rather than
            # raise, which is far harder to trace back than dropping it here.
            xmin, ymin = max(0.0, xmin), max(0.0, ymin)
            xmax, ymax = min(float(width), xmax), min(float(height), ymax)
            if xmax - xmin < 1.0 or ymax - ymin < 1.0:
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            if self.binary:
                labels.append(1)
            else:
                try:
                    labels.append(self.label_map[name])
                except KeyError:
                    raise KeyError(
                        f"Unknown annotation class {name!r} in "
                        f"{sample['xml_path']}; known classes: "
                        f"{sorted(self.label_map)}"
                    ) from None

        if boxes:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
        }

        img = F.to_tensor(img)
        if self.transforms:
            img, target = self.transforms(img, target)
        return img, target

    # ------------------------------------------------------------------ views

    def subset(self, predicate):
        """A shallow view holding only samples satisfying `predicate`.

        Used for per-condition and negatives-only evaluation without re-reading
        the annotations.
        """
        view = DroneDataset.__new__(DroneDataset)
        view.__dict__.update(self.__dict__)
        view.samples = [s for s in self.samples if predicate(s)]
        view.n_positive = sum(1 for s in view.samples if not s["is_negative"])
        view.n_negative = len(view.samples) - view.n_positive
        return view

    def negatives_only(self):
        return self.subset(lambda s: s["is_negative"])

    def positives_only(self):
        return self.subset(lambda s: not s["is_negative"])

    def with_transforms(self, transforms):
        """A view of this dataset using different transforms."""
        view = self.subset(lambda s: True)
        view.transforms = transforms
        return view

    def describe(self):
        from collections import Counter
        countries = Counter(
            s["meta"].get("country", "?") for s in self.samples
            if not s["is_negative"])
        categories = Counter(
            s["meta"].get("category", "?") for s in self.samples
            if s["is_negative"])
        return {
            "split": self.split,
            "total": len(self.samples),
            "positive": self.n_positive,
            "negative": self.n_negative,
            "dropped_unlabelled": getattr(self, "n_dropped_unlabelled", 0),
            "by_country": dict(countries.most_common()),
            "negative_categories": dict(categories.most_common()),
        }
