"""Class map and training hyperparameters.

The label space is no longer hardcoded. It is written by the dataset builder to
`<root>/classes.json` and loaded from there, because the builder is what decides
which classes actually survive: a country with too few images to evaluate is
dropped rather than left to contribute a meaningless AP. Hardcoding the classes
here and dropping them there would put the two out of sync, and a label-space
mismatch in a detector is silent - it mislabels every prediction instead of
raising.

Label 0 is reserved by torchvision's Faster R-CNN for the background class, so
foreground classes MUST start at 1. They previously started at 0, which meant
every "Turkish Drones" box was handed to the model as background: the class
scored exactly 0.000 AP because it was untrainable, not because it was hard. Any
checkpoint trained before that fix uses the old numbering and is not compatible
with this map.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

BACKGROUND_LABEL = 0

# Current country dataset. Binary training supplies its own --root and loads
# that dataset's one-class label map.
DATASET_ROOT = Path("drone_dataset_country_v3")
LEGACY_DATASET_ROOT = Path("drone_dataset_3_test")

# Used only if the dataset is absent, so importing the module still works.
# Matches the old four-country space, with foreground labels starting at 1.
_FALLBACK_LABEL_MAP = {
    "China": 1,
    "Iran": 2,
    "Israel": 3,
    "Turkey": 4,
}


def load_label_map(root=DATASET_ROOT):
    """Read the label map the dataset builder wrote, or fall back.

    Returns `(label_map, scheme, source)`.
    """
    path = Path(root) / "classes.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return data["label_map"], data.get("scheme", "country"), str(path)
    return dict(_FALLBACK_LABEL_MAP), "country", "config fallback"


CLASS_MAP, LABEL_SCHEME, LABEL_MAP_SOURCE = load_label_map()
CLASS_NAMES = {v: k for k, v in CLASS_MAP.items()}
NUM_CLASSES = len(CLASS_MAP)

if set(CLASS_MAP.values()) and min(CLASS_MAP.values()) == BACKGROUND_LABEL:
    raise RuntimeError(
        "a foreground class was given label 0, which torchvision reserves for "
        "background; that class would be untrainable"
    )


def label_for_name(name):
    """Map an annotation class name to its integer label.

    Raises rather than defaulting. The original `CLASS_MAP.get(name, 0)` silently
    folded every unrecognized name into label 0 - torchvision's background class
    - so a typo or a taxonomy change would quietly delete training signal instead
    of failing.
    """
    try:
        return CLASS_MAP[name]
    except KeyError:
        raise KeyError(
            f"Unknown annotation class {name!r}. "
            f"Known classes ({LABEL_MAP_SOURCE}): {sorted(CLASS_MAP)}"
        ) from None


@dataclass
class TrainConfig:
    root: Path = DATASET_ROOT
    weights_dir: Path = Path("weights")

    backbone: str = "resnet50"
    pretrained: bool = True
    freeze_backbone_epochs: int = 5
    min_size: int = 480
    max_size: int = 800
    fpn_out_channels: int = 256

    # Measured on the 4 GB RTX 3050 this trains on, at min_size=480/max_size=800,
    # backbone fully unfrozen, fp32:
    #
    #   bs=2   548 ms/step   1.83 GB
    #   bs=4  6986 ms/step   4.29 GB   <- lands on the card's capacity, thrashes
    #
    # Batch 4 collapses to roughly 12x the per-step time as activations spill to
    # host memory. The old README note about "headroom for a larger batch" was
    # measured on smaller images and does not hold here. Keep 2 unless the card
    # changes.
    #
    # Treat these as order-of-magnitude, not precise: images are variable size, so
    # per-step time varies with whatever the batch happened to contain. The
    # bs=4 result is far enough outside the noise to act on; small differences
    # between neighbouring configs are not.
    batch_size: int = 2
    num_workers: int = 0
    epochs: int = 30

    # Mixed precision. The 3050 is Ampere, so fp16 runs on tensor cores. Measured
    # at batch 2 with 3 backbone stages trainable: 417 ms/step and 1.74 GB peak,
    # against 954 ms/step and 3.42 GB without - roughly half the time and half the
    # memory, which is also what the hardware predicts. Disable only when
    # debugging a suspected numerical problem.
    amp: bool = True

    # torchvision trains only the last 3 backbone stages by default, and there is
    # no reason to deviate: the early stem and layer1 learn generic edge filters
    # that ImageNet already provides, and unfreezing them costs memory and time
    # for no accuracy. The previous code unfroze every backbone parameter.
    trainable_backbone_layers: int = 3
    lr: float = 0.005
    momentum: float = 0.9
    weight_decay: float = 0.0005
    lr_scheduler_step: int = 10
    lr_scheduler_gamma: float = 0.1

    # Adverse-condition augmentation. Half the samples are left clear so the
    # detector does not trade away good-weather accuracy for bad-weather
    # robustness; see data/transforms.py.
    adverse_prob: float = 0.5
    color_jitter: bool = True

    # Box filtering. `difficult` marks two unrelated things in the same flag - the
    # detector was unsure, and the box is nearly full-frame - so training filters
    # on the underlying quantities instead.
    #
    # min_score = 0.25 is the auto-labeller's own accept threshold, and therefore
    # the operating point whose box quality was actually measured: mean IoU 0.652
    # and 81.2% at IoU>=0.5 against human annotations
    # (reports/autolabel_validation.txt). The 0.40 "confident" cutoff used when
    # writing the annotations was extra conservatism that was never validated, and
    # honouring it here discarded 1558 training images - 24% of them, worst on the
    # thinnest classes - because an image whose every box is flagged has no objects
    # left and is dropped entirely.
    #
    # max_area_frac = 0.75 still excludes near-full-frame boxes. That is the defect
    # the old dataset carried 118 of, and it is worth excluding regardless of how
    # confident the detector was.
    skip_difficult: bool = True      # only consulted when min_score is None
    min_score: float = 0.25
    max_area_frac: float = 0.75

    rpn_nms_thresh: float = 0.7
    rpn_fg_iou_thresh: float = 0.7
    rpn_bg_iou_thresh: float = 0.3
    rpn_batch_size_per_image: int = 256
    rpn_positive_fraction: float = 0.5
    rpn_pre_nms_top_n_train: int = 2000
    rpn_pre_nms_top_n_test: int = 1000
    rpn_post_nms_top_n_train: int = 2000
    rpn_post_nms_top_n_test: int = 1000

    roi_pool_size: int = 7
    roi_batch_size_per_image: int = 512
    roi_positive_fraction: float = 0.25
    score_thresh: float = 0.05
    nms_thresh: float = 0.5
    detections_per_img: int = 100
    fg_iou_thresh: float = 0.5
    bg_iou_thresh: float = 0.5

    save_every: int = 5
    resume_from: str = None


if __name__ == "__main__":
    print(f"label scheme : {LABEL_SCHEME}")
    print(f"loaded from  : {LABEL_MAP_SOURCE}")
    print(f"classes      : {NUM_CLASSES} (+1 background)")
    for name, label in sorted(CLASS_MAP.items(), key=lambda kv: kv[1]):
        print(f"  {label:3d}  {name}")
