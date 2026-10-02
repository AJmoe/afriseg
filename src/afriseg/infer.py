"""Sliding-window inference on a whole (brain-cropped) volume."""
from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def predict_probs(model, img_norm: np.ndarray, roi, device, amp: bool = False,
                  flip_tta: bool = False, overlap: float = 0.5) -> np.ndarray:
    """img_norm (C,X,Y,Z) normalised -> sigmoid probabilities (3,X,Y,Z)."""
    from monai.inferers import sliding_window_inference

    model.eval()
    x = torch.from_numpy(img_norm[None]).to(device)
    flips = [()]
    if flip_tta:
        flips = [(), (2,), (3,), (4,), (2, 3), (2, 4), (3, 4), (2, 3, 4)]
    acc = None
    for f in flips:
        xi = torch.flip(x, f) if f else x
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            logits = sliding_window_inference(xi, roi_size=roi, sw_batch_size=2, predictor=model,
                                              overlap=overlap, mode="gaussian")
        p = torch.sigmoid(logits.float())
        p = torch.flip(p, f) if f else p
        acc = p if acc is None else acc + p
    return (acc / len(flips))[0].cpu().numpy()


def to_binary_regions(probs: np.ndarray, thr: float = 0.5) -> np.ndarray:
    """Enforce nesting ET within TC within WT, as in the BraTS region hierarchy."""
    b = probs > thr
    b[1] &= b[0]
    b[2] &= b[1]
    return b.astype(np.uint8)
