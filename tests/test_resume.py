"""Verify training resumes across the backbone freeze/unfreeze boundary.

Run with:  python -m tests.test_resume

The interesting case is not "does a checkpoint load", it is "does it load into an
optimizer of a DIFFERENT SIZE". The number of parameters the optimizer holds
depends on whether the backbone is frozen, which depends on the epoch. A resume
that builds the optimizer before reading the epoch fails with:

    ValueError: loaded state dict contains a parameter group that doesn't match
    the size of optimizer's group

which is exactly what happened resuming a real run saved at epoch 2 (backbone
frozen) once freeze_backbone_epochs had been passed. An earlier save/load test
missed it by saving and loading in the same freeze state.

Checks the three transitions that matter: frozen -> frozen, frozen -> unfrozen,
and unfrozen -> unfrozen.
"""
import sys
import tempfile
from pathlib import Path

import torch

from config import CLASS_MAP, TrainConfig
from models.faster_rcnn import build_model
from train import (build_optimizer, load_checkpoint, save_checkpoint,
                   set_backbone_trainable)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = TrainConfig()
tmp = Path(tempfile.mkdtemp(prefix="resumetest_"))


def make(epoch):
    """Build model+optimizer configured the way `main` would for `epoch`."""
    model = build_model(len(CLASS_MAP), pretrained=False,
                        min_size=cfg.min_size, max_size=cfg.max_size).to(device)
    frozen = epoch <= cfg.freeze_backbone_epochs
    set_backbone_trainable(model, not frozen, cfg.trainable_backbone_layers)
    lr = cfg.lr if frozen else cfg.lr * 0.1
    opt = build_optimizer(model, cfg, lr)
    sch = torch.optim.lr_scheduler.StepLR(opt, cfg.lr_scheduler_step,
                                          cfg.lr_scheduler_gamma)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    return model, opt, sch, scaler, frozen


def n_params(opt):
    return sum(len(g["params"]) for g in opt.param_groups)


cases = [
    ("frozen -> frozen", 2, 3),
    ("frozen -> UNFROZEN (the bug)", 2, 6),
    ("unfrozen -> unfrozen", 7, 8),
]

results = []
for label, saved_at, resume_at in cases:
    # Save as a run at `saved_at` would.
    model, opt, sch, scaler, frozen_saved = make(saved_at)
    for _ in range(saved_at - 1):
        sch.step()
    ckpt = tmp / f"ck_{saved_at}.pth"
    save_checkpoint(ckpt, saved_at, model, opt, sch, 0.42, 0.42, scaler)

    # Reproduce main()'s resume path: peek at the epoch, configure the freeze
    # state for it, build the optimizer, then load.
    peek = torch.load(ckpt, map_location="cpu", weights_only=False)
    start_epoch = int(peek["epoch"]) + 1
    del peek

    model2, opt2, sch2, scaler2, frozen_resume = make(resume_at)
    saved_n, resume_n = n_params(opt), n_params(opt2)

    try:
        # Force the freeze state the resumed epoch implies, exactly as main does.
        frozen_target = resume_at <= cfg.freeze_backbone_epochs
        set_backbone_trainable(model2, not frozen_target,
                               cfg.trainable_backbone_layers)
        opt3 = build_optimizer(model2, cfg,
                               cfg.lr if frozen_target else cfg.lr * 0.1)
        sch3 = torch.optim.lr_scheduler.StepLR(opt3, cfg.lr_scheduler_step,
                                               cfg.lr_scheduler_gamma)
        # The real mismatch case: the checkpoint's optimizer was sized for
        # `saved_at`, the optimizer here is sized for `resume_at`.
        nxt, best = load_checkpoint(ckpt, model2, opt3, sch3, device, scaler2)
        # Loading must not raise, must restore the epoch, and must restore the
        # weights. Optimizer momentum is allowed to be dropped when the parameter
        # sets differ - see load_checkpoint - so it is not asserted here.
        ok = (nxt == start_epoch)
        same_weights = all(
            torch.equal(a.detach().cpu(), b.detach().cpu())
            for a, b in zip(model.parameters(), model2.parameters()))
        ok = ok and same_weights
        detail = (f"saved_opt={saved_n} params, resume_opt={n_params(opt3)} params, "
                  f"next_epoch={nxt}, weights_restored={same_weights}")
    except Exception as exc:                                       # noqa: BLE001
        ok = False
        detail = f"{type(exc).__name__}: {exc}"

    results.append((label, ok, detail))
    del model, model2
    if device.type == "cuda":
        torch.cuda.empty_cache()

print("\n================ CHECKS ================")
all_ok = True
for label, ok, detail in results:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    print(f"        {detail}")
    if not ok:
        all_ok = False

import shutil
shutil.rmtree(tmp, ignore_errors=True)
print("\n" + ("ALL RESUME CHECKS PASSED" if all_ok else "SOME CHECKS FAILED"))
sys.exit(0 if all_ok else 1)
