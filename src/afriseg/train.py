"""Train one experimental arm.

Source-only (train on BraTS 2021 holdout train split, pick best epoch on its val split):
  python -m afriseg.train --manifest npz/brats2021_manifest.csv --split splits/brats2021.json \
      --subset train --val-subset val --aug 15t --out runs/A2_15t

Fine-tune on African cases, k-fold (fold 0 is held out; NO checkpoint selection on it):
  python -m afriseg.train --manifest npz/africa_manifest.csv --split splits/africa.json \
      --subset 0 --exclude --n-labelled 10 --init runs/A2_15t/best.pt --lr 1e-4 \
      --aug 15t --epochs 50 --out runs/B_15t_k10_f0
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .aug15t import Aug15TConfig
from .dataset import PatchDataset, load_npz, select_rows
from .infer import predict_probs, to_binary_regions
from .labels import REGIONS, to_regions
from .metrics import region_scores
from .model import build_model
from .transforms import TrainTransform, eval_transform


def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split")
    ap.add_argument("--subset", default="train", help="split value used for training (holdout name or fold index)")
    ap.add_argument("--exclude", action="store_true", help="train on every case EXCEPT --subset (k-fold)")
    ap.add_argument("--val-subset", help="holdout split value for model selection (e.g. val)")
    ap.add_argument("--n-labelled", type=int)
    ap.add_argument("--aug", default="none")
    ap.add_argument("--aug-config", help="JSON file with Aug15TConfig overrides (calibrated ranges)")
    ap.add_argument("--model", default="segresnet")
    ap.add_argument("--small", action="store_true", help="tiny network for smoke tests")
    ap.add_argument("--init", help="checkpoint to initialise from (fine-tuning)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--iters", type=int, default=250, help="patches per epoch")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--patch", type=int, nargs=3, default=(128, 128, 128))
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--val-every", type=int, default=10)
    ap.add_argument("--val-max-cases", type=int, default=40)
    ap.add_argument("--max-hours", type=float, default=11.5, help="stop and checkpoint before Kaggle's 12 h limit")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", required=True)
    return ap.parse_args(argv)


def loss_fn():
    from monai.losses import DiceLoss

    dice = DiceLoss(sigmoid=True, smooth_nr=0.0, smooth_dr=1e-5, squared_pred=True)
    bce = torch.nn.BCEWithLogitsLoss()
    return lambda logits, y: dice(logits, y) + bce(logits, y)


def validate(model, rows, roi, device, amp) -> dict:
    scores = []
    for r in rows:
        img, lbl = load_npz(r["path"])
        probs = predict_probs(model, eval_transform(img), roi, device, amp)
        scores.append(region_scores(to_binary_regions(probs), to_regions(lbl), with_hd=False))
    out = {f"dice_{k}": float(np.mean([s[f"dice_{k}"] for s in scores])) for k in REGIONS}
    out["dice_mean"] = float(np.mean([out[f"dice_{k}"] for k in REGIONS]))
    return out


def main(argv=None):
    a = get_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device)
    amp = device.type == "cuda"

    cfg = Aug15TConfig.from_json(a.aug_config) if a.aug_config else Aug15TConfig()
    train_rows = select_rows(a.manifest, a.split, a.subset, a.exclude, a.n_labelled, a.seed)
    val_rows = select_rows(a.manifest, a.split, a.val_subset) if a.val_subset else []
    val_rows = val_rows[: a.val_max_cases]
    select_best = bool(val_rows)
    (out / "args.json").write_text(json.dumps(vars(a), indent=1))
    (out / "train_ids.json").write_text(json.dumps(sorted(r["id"] for r in train_rows)))
    cfg.to_json(out / "aug15t_config.json")
    print(f"device={device} train_cases={len(train_rows)} val_cases={len(val_rows)} aug={a.aug}")

    ds = PatchDataset(train_rows, TrainTransform(a.aug, a.patch, cfg), a.iters * a.batch)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, num_workers=a.workers,
                                     pin_memory=amp, persistent_workers=a.workers > 0, drop_last=True)
    model = build_model(a.model, small=a.small).to(device)
    if a.init:
        state = torch.load(a.init, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        print("initialised from", a.init)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda e: 0.5 * (1 + math.cos(math.pi * min(e, a.epochs) / a.epochs)))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    crit = loss_fn()
    start_epoch, best = 0, -1.0

    last = out / "last.pt"
    if a.resume and last.exists():
        st = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"]); scaler.load_state_dict(st["scaler"])
        start_epoch, best = st["epoch"] + 1, st["best"]
        print(f"resumed at epoch {start_epoch}")

    def save(path, epoch):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "best": best, "args": vars(a)}, path)

    t0 = time.time()
    log = open(out / "log.jsonl", "a")
    for epoch in range(start_epoch, a.epochs):
        model.train()
        losses, te = [], time.time()
        for batch in dl:
            x, y = batch["image"].to(device), batch["label"].to(device).float()
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.item()))
        sched.step()
        rec = {"epoch": epoch, "loss": float(np.mean(losses)), "lr": opt.param_groups[0]["lr"],
               "sec": round(time.time() - te, 1)}
        if select_best and ((epoch + 1) % a.val_every == 0 or epoch + 1 == a.epochs):
            v = validate(model, val_rows, a.patch, device, amp)
            rec.update(v)
            if v["dice_mean"] > best:
                best = v["dice_mean"]
                save(out / "best.pt", epoch)
        print(json.dumps(rec))
        log.write(json.dumps(rec) + "\n"); log.flush()
        save(last, epoch)
        if (time.time() - t0) / 3600 > a.max_hours and epoch + 1 < a.epochs:
            print(f"Time budget reached after epoch {epoch}; rerun with --resume to continue.")
            break
    else:
        # k-fold / no validation set: the final model is THE model (no selection on test data)
        if not select_best:
            save(out / "best.pt", a.epochs - 1)
        (out / "DONE").write_text("ok")
    log.close()


if __name__ == "__main__":
    main()
