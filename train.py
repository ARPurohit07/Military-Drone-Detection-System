"""Train the detector on the rebuilt dataset.

Three things changed here beyond pointing at the new dataset.

Resume actually works. `TrainConfig.resume_from` was declared but never read, so
the previous retrain - interrupted after epoch 6 of 30 - had no way to continue
and the work was lost. Checkpoints now carry the epoch, optimizer and scheduler
state, and `--resume` restores all of it.

Adverse-condition augmentation is on by default, from `TrainConfig.adverse_prob`.
That is what addresses the project's third goal; see data/transforms.py for why
it happens here rather than as files on disk.

Negatives are part of the split, not bolted on. The dataset builder assigns each
one to train or val, so there is no longer a flag that loads every negative into
both - which previously meant the false-positive rate was measured on images the
model had trained on.
"""

import argparse
import gc
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (CLASS_MAP, LABEL_MAP_SOURCE, LABEL_SCHEME, TrainConfig,
                    load_label_map)
from data.dataset import DroneDataset
from data.transforms import (get_train_transforms, get_val_transforms,
                             set_transform_seed)
from models.faster_rcnn import build_model


def collate_fn(batch):
    return tuple(zip(*batch))


class NullWriter:
    """TensorBoard-compatible no-op for memory-constrained training."""

    def add_scalar(self, *args, **kwargs):
        pass

    def close(self):
        pass


def build_class_balanced_sampler(dataset, power=0.5):
    """Sampler that draws rare classes more often, for imbalanced label spaces.

    The registry is deliberately deeper for well-documented countries, so Turkey
    and Iran collect ~1000 images while Italy, Germany and South Korea collect
    ~200-280. Left alone, the loss is dominated by the large classes and the thin
    ones are effectively ignored.

    `power` controls how aggressively that is corrected. Weight is
    ``(1 / class_count) ** power``, so 0 is no correction, 1.0 is full inverse
    frequency, and the 0.5 default is the usual square-root compromise: full
    inversion makes a 250-image class appear as often as a 1000-image one, which
    means each of its images is seen four times per epoch and it overfits on those
    specific images instead. Square root closes most of the gap without that.

    Negatives are weighted as one additional group rather than per category, so
    hard negatives keep their intended share of the batch.
    """
    from collections import Counter

    keys = []
    for sample in dataset.samples:
        if sample["is_negative"]:
            keys.append("__negative__")
        else:
            keys.append(sample["meta"].get("country")
                        or sample["meta"].get("airframe") or "__unknown__")

    counts = Counter(keys)
    weights = [(1.0 / counts[k]) ** power for k in keys]
    return torch.utils.data.WeightedRandomSampler(
        weights, num_samples=len(weights), replacement=True), counts


def build_binary_hard_negative_sampler(dataset, num_samples=None,
                                        focus_category=None,
                                        focus_mass=0.0):
    """Allocate half an epoch to positives and half across negative categories.

    Newly mined false-positive categories are necessarily much smaller than the
    original corpus. Uniform shuffling would show a 178-image flock collection
    in under one percent of batches, which is too weak to move an already-trained
    detector. This sampler preserves a 50/50 positive-negative split and divides
    the negative half evenly by category without duplicating files on disk.
    """
    from collections import Counter

    keys = []
    for sample in dataset.samples:
        if sample["is_negative"]:
            category = sample["meta"].get("category") or "other"
            keys.append(f"negative:{category}")
        else:
            keys.append("positive")
    counts = Counter(keys)
    negative_keys = sorted(key for key in counts if key.startswith("negative:"))
    if not negative_keys or not counts.get("positive"):
        raise ValueError("hard-negative balancing needs positives and negatives")
    focus_key = (f"negative:{focus_category}" if focus_category else None)
    if focus_key and focus_key not in counts:
        raise ValueError(
            f"focused negative category {focus_category!r} is not in the dataset"
        )
    if not 0.0 <= focus_mass < 0.5:
        raise ValueError("focus_mass must be in [0, 0.5)")

    # Keep half the epoch positive so hard-negative adaptation cannot crowd out
    # the detector's actual task. A focused category can claim part of the
    # negative half; all remaining categories share the rest evenly.
    ordinary_negative_keys = [key for key in negative_keys if key != focus_key]
    ordinary_mass = 0.5 - (focus_mass if focus_key else 0.0)
    negative_group_mass = ordinary_mass / len(ordinary_negative_keys)
    weights = []
    for key in keys:
        if key == "positive":
            weights.append(0.5 / counts[key])
        elif key == focus_key:
            weights.append(focus_mass / counts[key])
        else:
            weights.append(negative_group_mass / counts[key])
    return torch.utils.data.WeightedRandomSampler(
        weights, num_samples=num_samples or len(weights), replacement=True), counts


# Backbone stages, outermost first. ResNet-FPN names them body.layer1..layer4
# plus the stem (conv1/bn1). Unfreezing is applied from the deep end inward,
# which is the standard recipe: deep stages carry task-specific semantics, the
# stem carries generic edge filters that ImageNet already supplies.
_BACKBONE_STAGES = ("layer4", "layer3", "layer2", "layer1", "conv1", "bn1")


def set_backbone_trainable(model, trainable, trainable_layers=3):
    """Freeze or unfreeze the backbone.

    `trainable=False` freezes all of it. `trainable=True` unfreezes only the last
    `trainable_layers` stages, matching torchvision's own default rather than
    unfreezing everything - which is what the previous version did, at a real
    cost in memory and step time for no accuracy gain.
    """
    unfrozen = set(_BACKBONE_STAGES[:trainable_layers]) if trainable else set()
    for name, param in model.named_parameters():
        if "backbone" not in name:
            continue
        param.requires_grad = any(stage in name for stage in unfrozen)


def build_optimizer(model, cfg, lr):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.SGD(params, lr=lr, momentum=cfg.momentum,
                           weight_decay=cfg.weight_decay)


def train_one_epoch(model, optimizer, loader, device, epoch, writer,
                    scaler=None, checkpoint_callback=None,
                    checkpoint_every=0, start_batch=0):
    model.train()
    total_loss = 0.0
    processed = 0
    if start_batch and loader.num_workers == 0:
        # Fast-forward indices without decoding/augmenting completed samples.
        # The project's loader intentionally uses zero workers on Windows, so
        # reconstructing the remaining batches here preserves sampler order and
        # turns a 7-minute 9k-batch replay into a sub-second iterator advance.
        sampler = iter(loader.batch_sampler)
        for _ in range(min(start_batch, len(loader))):
            next(sampler)

        def remaining_batches():
            for indices in sampler:
                yield loader.collate_fn([loader.dataset[i] for i in indices])

        pbar = tqdm(remaining_batches(), total=len(loader),
                    initial=start_batch, desc=f"Epoch {epoch}")
        batches = enumerate(pbar, start=start_batch)
        print(f"  resumed at batch {start_batch}/{len(loader)}")
    else:
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        batches = enumerate(pbar)

    for batch_idx, (images, targets) in batches:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        with torch.autocast("cuda", dtype=torch.float16,
                            enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        if scaler is not None:
            scaler.scale(losses).backward()
            # Unscale before clipping, or the clip threshold is applied to
            # gradients that are still multiplied by the loss scale and does
            # effectively nothing.
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += losses.item()
        processed += 1
        global_step = (epoch - 1) * len(loader) + batch_idx
        writer.add_scalar("train/loss_total", losses.item(), global_step)
        pbar.set_postfix(loss=f"{losses.item():.4f}")

        if batch_idx % 50 == 0:
            for k, v in loss_dict.items():
                writer.add_scalar(f"train/{k}", v.item(), global_step)

        # Release Python references, but leave freed blocks in PyTorch's caching
        # allocator. Calling empty_cache every few batches forced thousands of
        # cudaFree/cudaMalloc cycles and produced a repeatable native c10.dll
        # access violation under WDDM on this 4 GB card.
        del images, targets, loss_dict, losses
        if (checkpoint_callback is not None and checkpoint_every
                and (batch_idx + 1) % checkpoint_every == 0):
            checkpoint_callback(batch_idx + 1)

    return total_loss / max(1, processed)


@torch.no_grad()
def validate(model, loader, device, amp=False):
    """Validation loss.

    torchvision's detection models only return losses in train mode, so the mode
    is left as train and gradients are disabled by the decorator instead. This
    measures loss, not detection quality - use evaluate.py for mAP.
    """
    model.train()
    total_loss = 0.0
    for batch_idx, (images, targets) in enumerate(loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            loss_dict = model(images, targets)
        total_loss += sum(loss for loss in loss_dict.values()).item()
        del images, targets, loss_dict
    return total_loss / max(1, len(loader))


# Set by main() so a checkpoint records the label space it was actually trained
# on - a binary detector loaded against the 14-class map would mislabel silently.
CLASS_MAP_ACTIVE = CLASS_MAP
LABEL_SCHEME_ACTIVE = LABEL_SCHEME


def save_checkpoint(path, epoch, model, optimizer, scheduler, val_loss,
                    best_val_loss, scaler=None, seed=1337,
                    batch_in_epoch=0):
    """Write a checkpoint that is sufficient to resume from.

    The old code saved only `model.state_dict()` for the best model, which is
    enough to evaluate but not to continue: restarting reset the optimizer
    momentum, the LR schedule and the epoch counter.
    """
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "class_map": CLASS_MAP_ACTIVE,
        "label_scheme": LABEL_SCHEME_ACTIVE,
        "seed": seed,
        "batch_in_epoch": batch_in_epoch,
        "model_config": {
            "min_size": int(model.transform.min_size[0]),
            "max_size": int(model.transform.max_size),
        },
    }, temporary)
    # A native CUDA crash cannot leave the last known-good checkpoint half
    # overwritten: replacement is atomic when both files share a directory.
    os.replace(temporary, path)


def warm_start_backbone(path, model, device):
    """Load a checkpoint's weights except the class-dependent box predictor.

    Used to start the single-class detector from the 14-class one. Backbone, FPN
    and RPN are all class-agnostic and represent nearly all of the ~37M trained
    parameters; only `roi_heads.box_predictor` is shaped by the number of classes
    (~20k parameters) and has to be discarded.

    That makes changing the label space cheap: the expensive part - features that
    already know what a drone looks like - carries over, so the new head converges
    in a handful of epochs instead of the 15 the original run needed from COCO.
    """
    blob = torch.load(path, map_location=device, weights_only=False)
    state = blob.get("model_state_dict", blob) if isinstance(blob, dict) else blob

    own = model.state_dict()
    loaded, skipped = [], []
    for k, v in state.items():
        if k in own and own[k].shape == v.shape:
            own[k] = v
            loaded.append(k)
        else:
            skipped.append(k)

    # If both checkpoints describe named label spaces, transfer predictor rows
    # by class name rather than by numeric id. This preserves learned country
    # weights when the taxonomy expands/reorders, while newly added countries
    # keep their random initialisation. Background is always row 0.
    saved_map = blob.get("class_map") if isinstance(blob, dict) else None
    transferred = []
    if saved_map and saved_map != CLASS_MAP_ACTIVE:
        score_weight = "roi_heads.box_predictor.cls_score.weight"
        score_bias = "roi_heads.box_predictor.cls_score.bias"
        box_weight = "roi_heads.box_predictor.bbox_pred.weight"
        box_bias = "roi_heads.box_predictor.bbox_pred.bias"
        if all(key in own and key in state
               for key in (score_weight, score_bias, box_weight, box_bias)):
            own[score_weight][0] = state[score_weight][0]
            own[score_bias][0] = state[score_bias][0]
            own[box_weight][0:4] = state[box_weight][0:4]
            own[box_bias][0:4] = state[box_bias][0:4]
            for name in sorted(set(saved_map) & set(CLASS_MAP_ACTIVE)):
                source_id = int(saved_map[name])
                target_id = int(CLASS_MAP_ACTIVE[name])
                own[score_weight][target_id] = state[score_weight][source_id]
                own[score_bias][target_id] = state[score_bias][source_id]
                own[box_weight][4 * target_id:4 * (target_id + 1)] = (
                    state[box_weight][4 * source_id:4 * (source_id + 1)])
                own[box_bias][4 * target_id:4 * (target_id + 1)] = (
                    state[box_bias][4 * source_id:4 * (source_id + 1)])
                transferred.append(name)
    model.load_state_dict(own)
    print(f"  warm start from {path}: {len(loaded)} tensors loaded, "
          f"{len(skipped)} skipped (class-dependent head)")
    if skipped:
        print(f"    skipped: {', '.join(sorted(skipped)[:4])}"
              f"{' ...' if len(skipped) > 4 else ''}")
    if transferred:
        print("    semantically transferred predictor rows: "
              + ", ".join(transferred))
    return len(loaded)


def load_checkpoint(path, model, optimizer, scheduler, device, scaler=None):
    """Restore training state. Returns (next_epoch, best_val_loss)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "model_state_dict" not in ckpt:
        # A bare state_dict from the old save path: weights only, no history.
        model.load_state_dict(ckpt)
        print(f"  loaded weights only from {path} (no optimizer/epoch state)")
        return 1, float("inf")

    # Compare against the label space this run is ACTUALLY training, not the
    # dataset's default. A --binary run trains {"drone": 1} while config.CLASS_MAP
    # still holds the dataset's country classes, so checking the latter made a binary
    # checkpoint impossible to resume from.
    saved_map = ckpt.get("class_map")
    if saved_map and saved_map != CLASS_MAP_ACTIVE:
        raise SystemExit(
            f"checkpoint label space does not match the current one.\n"
            f"  checkpoint: {saved_map}\n"
            f"  current   : {CLASS_MAP_ACTIVE}\n"
            f"Loading it would mislabel every prediction silently."
        )

    model.load_state_dict(ckpt["model_state_dict"])
    if ckpt.get("optimizer_state_dict"):
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except ValueError as exc:
            # The optimizer holds a different number of parameters than the
            # checkpoint did, which happens when a run saved with the backbone
            # frozen is resumed after it unfreezes (14 parameter tensors vs 140).
            # There is no meaningful way to transfer momentum for parameters that
            # were not being optimised, so the weights are kept and the optimiser
            # restarts. Losing momentum costs a little progress in the first
            # epoch after the resume; crashing costs the whole run.
            print(f"  optimizer state not restored ({exc}).")
            print("  This is expected when resuming across the backbone "
                  "freeze/unfreeze boundary; weights loaded, momentum restarts.")
    if ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler is not None and ckpt.get("scaler_state_dict"):
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    epoch = int(ckpt.get("epoch", 0))
    best = float(ckpt.get("best_val_loss", ckpt.get("val_loss", float("inf"))))
    print(f"  resumed from {path} at epoch {epoch} (best val_loss {best:.4f})")
    return epoch + 1, best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=str, default=None,
                        help="checkpoint to resume from")
    parser.add_argument("--reset-best", action="store_true",
                        help="when resuming on a changed dataset, reset the "
                             "best validation loss so the new run saves its "
                             "own best checkpoint")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--min-size", type=int, default=None,
                        help="Faster R-CNN internal short-side resize")
    parser.add_argument("--max-size", type=int, default=None,
                        help="Faster R-CNN internal long-side cap")
    parser.add_argument("--adverse-prob", type=float, default=None,
                        help="probability of an adverse-condition augmentation")
    parser.add_argument("--no-amp", action="store_true",
                        help="disable mixed precision (slower; for debugging)")
    parser.add_argument("--binary", action="store_true",
                        help="train a single-class drone detector (stage 1 of "
                             "the two-stage pipeline)")
    parser.add_argument("--warm-start", type=str, default=None,
                        help="load backbone/FPN/RPN from this checkpoint, "
                             "skipping the class-dependent box predictor")
    parser.add_argument("--balance-classes", type=float, default=None,
                        metavar="POWER",
                        help="draw rare classes more often; 0.5 is the usual "
                             "square-root compromise, 1.0 full inverse frequency")
    parser.add_argument(
        "--balance-negative-categories", action="store_true",
        help="binary hard-negative mining: sample 50%% positives and divide "
             "the negative half evenly across negative categories",
    )
    parser.add_argument(
        "--samples-per-epoch", type=int, default=None,
        help="number of weighted samples per epoch; useful for focused hard-"
             "negative fine-tuning without replaying the entire corpus",
    )
    parser.add_argument(
        "--focus-negative-category", type=str, default=None,
        help="negative category to emphasize during binary hard-negative mining",
    )
    parser.add_argument(
        "--focus-negative-mass", type=float, default=0.0,
        help="fraction of all samples assigned to the focused negative category; "
             "must be below 0.5 (example: 0.35)",
    )
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--weights-dir", type=str, default=None,
                        help="run-specific checkpoint directory; use separate "
                             "directories for binary and country models")
    parser.add_argument("--run-name", type=str, default=None,
                        help="TensorBoard run name (default: binary or country)")
    parser.add_argument("--step-checkpoint-every", type=int, default=1000,
                        help="atomically save in-epoch recovery state every N "
                             "batches; 0 disables it")
    parser.add_argument("--no-tensorboard", action="store_true",
                        help="avoid loading TensorBoard/TensorFlow into RAM")
    parser.add_argument("--seed", type=int, default=1337,
                        help="best-effort training seed (default: 1337)")
    parser.add_argument("--resume-batch", type=int, default=None,
                        help="override in-epoch batch position for an older "
                             "recovery checkpoint")
    parser.add_argument("--skip-val-loss", action="store_true",
                        help="save the completed epoch without validation-loss "
                             "selection; use held-out mAP/FPR for selection")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    set_transform_seed(args.seed)

    cfg = TrainConfig()
    if args.epochs:
        cfg.epochs = args.epochs
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.min_size:
        cfg.min_size = args.min_size
    if args.max_size:
        cfg.max_size = args.max_size
    if args.adverse_prob is not None:
        cfg.adverse_prob = args.adverse_prob
    if args.root:
        cfg.root = Path(args.root)
    elif args.binary:
        cfg.root = Path("drone_dataset_binary_v4_birds")
    if args.weights_dir:
        cfg.weights_dir = Path(args.weights_dir)
    if args.no_amp:
        cfg.amp = False
    resume_from = args.resume or cfg.resume_from

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Dataset: {cfg.root}")
    # In binary mode the label space is a single "drone" class regardless of what
    # the dataset's classes.json says; country is a separate model downstream.
    if args.binary:
        active_map, scheme, map_source = ({"drone": 1}, "drone (binary)",
                                          "binary mode")
    else:
        active_map, scheme, map_source = load_label_map(cfg.root)
    globals()["CLASS_MAP_ACTIVE"] = active_map
    globals()["LABEL_SCHEME_ACTIVE"] = scheme
    print(f"Label scheme: {scheme} ({len(active_map)} classes, "
          f"from {map_source})")

    os.makedirs(cfg.weights_dir, exist_ok=True)

    train_dataset = DroneDataset(
        cfg.root, "train",
        get_train_transforms(adverse_prob=cfg.adverse_prob,
                             jitter=cfg.color_jitter),
        skip_difficult=cfg.skip_difficult,
        min_score=cfg.min_score, max_area_frac=cfg.max_area_frac,
        binary=args.binary, label_map=active_map,
    )
    val_dataset = DroneDataset(
        cfg.root, "val", get_val_transforms(),
        skip_difficult=cfg.skip_difficult,
        min_score=cfg.min_score, max_area_frac=cfg.max_area_frac,
        binary=args.binary, label_map=active_map,
    )
    print(f"Train: {len(train_dataset)} "
          f"({train_dataset.n_positive} with drones, "
          f"{train_dataset.n_negative} negatives)")
    print(f"Val  : {len(val_dataset)} "
          f"({val_dataset.n_positive} with drones, "
          f"{val_dataset.n_negative} negatives)")
    print(f"Adverse-condition augmentation: p={cfg.adverse_prob}")
    print(f"Box filter: min_score={cfg.min_score}, "
          f"max_area_frac={cfg.max_area_frac} "
          f"({train_dataset.n_dropped_unlabelled} train images had no box left "
          f"and were dropped)")

    sampler = None
    if args.balance_negative_categories:
        if not args.binary:
            raise SystemExit("--balance-negative-categories requires --binary")
        sampler, class_counts = build_binary_hard_negative_sampler(
            train_dataset, num_samples=args.samples_per_epoch,
            focus_category=args.focus_negative_category,
            focus_mass=args.focus_negative_mass,
        )
        detail = ""
        if args.focus_negative_category:
            detail = (f", {args.focus_negative_mass:.0%} "
                      f"{args.focus_negative_category}")
        print(f"Binary hard-negative category balancing (50% positive{detail}):")
        for name, n in sorted(class_counts.items(), key=lambda kv: -kv[1]):
            print(f"    {n:6d}  {name}")
    elif args.balance_classes is not None:
        sampler, class_counts = build_class_balanced_sampler(
            train_dataset, power=args.balance_classes)
        print(f"Class-balanced sampling (power={args.balance_classes}):")
        for name, n in sorted(class_counts.items(), key=lambda kv: -kv[1]):
            print(f"    {n:6d}  {name}")

    # Windows/WDDM intermittently fails pinned host mappings after a long CUDA
    # run ("CUDA error: resource already mapped"). With num_workers=0 there is
    # no overlap benefit worth that instability, so pin only on other systems.
    pin_memory = device.type == "cuda" and os.name != "nt"
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size,
        # A sampler and shuffle are mutually exclusive in DataLoader.
        shuffle=(sampler is None), sampler=sampler,
        num_workers=cfg.num_workers, collate_fn=collate_fn,
        pin_memory=pin_memory, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=collate_fn,
        pin_memory=pin_memory,
    )

    model = build_model(
        num_classes=len(active_map), pretrained=cfg.pretrained,
        min_size=cfg.min_size, max_size=cfg.max_size,
    ).to(device)
    if args.warm_start:
        if not Path(args.warm_start).exists():
            raise SystemExit(f"no such checkpoint: {args.warm_start}")
        warm_start_backbone(args.warm_start, model, device)

    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    print(f"Mixed precision: {'on' if use_amp else 'off'}")

    # The epoch has to be known BEFORE the optimizer is constructed. How many
    # parameters the optimizer holds depends on whether the backbone is frozen,
    # which depends on the epoch - so building it first and loading a checkpoint
    # into it afterwards fails with "a parameter group that doesn't match the size
    # of optimizer's group" whenever the run is resumed into a different freeze
    # phase than it was saved in. Peek at the epoch, set the freeze state for it,
    # and only then build the optimizer that the state will be loaded into.
    start_epoch, best_val_loss = 1, float("inf")
    resume_batch = args.resume_batch or 0
    if resume_from:
        if not Path(resume_from).exists():
            raise SystemExit(f"no such checkpoint: {resume_from}")
        peek = torch.load(resume_from, map_location="cpu", weights_only=False)
        if isinstance(peek, dict) and "epoch" in peek:
            start_epoch = int(peek["epoch"]) + 1
            if args.resume_batch is None:
                resume_batch = int(peek.get("batch_in_epoch", 0))
        del peek

    want_frozen = start_epoch <= cfg.freeze_backbone_epochs
    set_backbone_trainable(model, not want_frozen, cfg.trainable_backbone_layers)
    lr = cfg.lr if want_frozen else cfg.lr * 0.1
    optimizer = build_optimizer(model, cfg, lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.lr_scheduler_step,
        gamma=cfg.lr_scheduler_gamma,
    )
    for _ in range(start_epoch - 1):
        scheduler.step()

    if resume_from:
        start_epoch, best_val_loss = load_checkpoint(
            resume_from, model, optimizer, scheduler, device, scaler)
        if args.reset_best:
            best_val_loss = float("inf")
            print("  reset best validation loss for the new dataset")

    run_name = args.run_name or ("binary" if args.binary else "country")
    if args.no_tensorboard:
        writer = NullWriter()
        print("TensorBoard logging: off")
    else:
        # Import lazily: the installed TensorBoard stack also imports
        # TensorFlow, which consumes several GB of host commit even though this
        # training loop itself is PyTorch-only.
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(Path("runs") / run_name))
    # Already configured for start_epoch above, so the loop must not rebuild the
    # optimizer on its first pass and discard the state just restored.
    backbone_frozen = want_frozen
    state = "frozen" if want_frozen else "unfrozen"
    print(f"  backbone {state}, lr={lr}, starting at epoch {start_epoch}")

    for epoch in range(start_epoch, cfg.epochs + 1):
        # Recomputed every epoch rather than only at the boundary, so a resume
        # into the unfrozen phase does not silently train with a frozen backbone.
        want_frozen = epoch <= cfg.freeze_backbone_epochs
        if want_frozen != backbone_frozen:
            set_backbone_trainable(model, not want_frozen,
                                   cfg.trainable_backbone_layers)
            lr = cfg.lr if want_frozen else cfg.lr * 0.1
            optimizer = build_optimizer(model, cfg, lr)
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=cfg.lr_scheduler_step,
                gamma=cfg.lr_scheduler_gamma,
            )
            # Advance the schedule to where this epoch should be.
            for _ in range(epoch - 1):
                scheduler.step()
            backbone_frozen = want_frozen
            state = "frozen" if want_frozen else "unfrozen"
            print(f"  backbone {state}, lr={lr}")

        def save_progress(batch_number):
            # Store epoch-1 so resume re-enters this epoch. It may repeat up to
            # N-1 shuffled batches, which is preferable to losing the full run.
            progress_path = cfg.weights_dir / "progress_checkpoint.pth"
            save_checkpoint(progress_path, epoch - 1, model, optimizer,
                            scheduler, float("nan"), best_val_loss, scaler,
                            args.seed, batch_in_epoch=batch_number)
            # Variable image sizes make CUDA's caching allocator retain blocks
            # that Windows/WDDM also charges against the system commit limit.
            # Release only at checkpoint boundaries: if the driver dislikes a
            # cache flush, the just-written atomic checkpoint is recoverable.
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"\n  -> recovery checkpoint after batch {batch_number}: "
                  f"{progress_path}")

        train_loss = train_one_epoch(
            model, optimizer, train_loader, device, epoch, writer, scaler,
            checkpoint_callback=save_progress,
            checkpoint_every=max(0, args.step_checkpoint_every),
            start_batch=resume_batch if epoch == start_epoch else 0)
        resume_batch = 0

        # Persist all completed updates before validation. A native CUDA fault
        # in validation can then resume at len(train_loader) with no replay.
        pre_validation = cfg.weights_dir / "pre_validation_checkpoint.pth"
        save_checkpoint(pre_validation, epoch - 1, model, optimizer,
                        scheduler, float("nan"), best_val_loss, scaler,
                        args.seed, batch_in_epoch=len(train_loader))

        if args.skip_val_loss:
            val_loss = float("nan")
            print("Validation loss skipped; select this checkpoint with "
                  "held-out mAP/FPR.")
        else:
            val_loss = validate(model, val_loader, device, use_amp)
        scheduler.step()

        writer.add_scalar("val/loss_total", val_loss, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, "
              f"val_loss={val_loss:.4f}")

        if args.skip_val_loss or val_loss < best_val_loss:
            if not args.skip_val_loss:
                best_val_loss = val_loss
            save_checkpoint(cfg.weights_dir / "best_model.pth", epoch, model,
                            optimizer, scheduler, val_loss, best_val_loss,
                            scaler, args.seed)
            print(f"  -> saved best model (val_loss={val_loss:.4f})")

        # Always keep a resumable latest, not only every `save_every` epochs:
        # the previous run lost 6 epochs because the interruption fell between
        # two periodic saves.
        save_checkpoint(cfg.weights_dir / "last_checkpoint.pth", epoch, model,
                        optimizer, scheduler, val_loss, best_val_loss, scaler,
                        args.seed)
        if epoch % cfg.save_every == 0:
            save_checkpoint(cfg.weights_dir / f"checkpoint_epoch{epoch}.pth",
                            epoch, model, optimizer, scheduler, val_loss,
                            best_val_loss, scaler, args.seed)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    writer.close()
    print("Training complete.")
    print(f"Best val_loss: {best_val_loss:.4f}")
    print("\nEvaluate with:")
    print(f"  python evaluate.py --weights {cfg.weights_dir / 'best_model.pth'} "
          f"{'--binary ' if args.binary else ''}--adverse")


if __name__ == "__main__":
    main()
