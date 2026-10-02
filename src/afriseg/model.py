"""Network factory. SegResNet is the primary backbone (fits a 16 GB Kaggle GPU at 128^3)."""
from __future__ import annotations

import torch.nn as nn


def build_model(name: str = "segresnet", in_channels: int = 4, out_channels: int = 3,
                small: bool = False) -> nn.Module:
    if name == "segresnet":
        from monai.networks.nets import SegResNet
        return SegResNet(
            spatial_dims=3, in_channels=in_channels, out_channels=out_channels,
            init_filters=8 if small else 16,
            blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1), dropout_prob=0.2,
        )
    if name == "unet":
        from monai.networks.nets import UNet
        ch = (8, 16, 32, 64) if small else (32, 64, 128, 256, 320)
        return UNet(spatial_dims=3, in_channels=in_channels, out_channels=out_channels,
                    channels=ch, strides=(2,) * (len(ch) - 1), num_res_units=2, norm="instance")
    if name == "swinunetr":
        from monai.networks.nets import SwinUNETR
        return SwinUNETR(in_channels=in_channels, out_channels=out_channels,
                         feature_size=12 if small else 48, spatial_dims=3)
    raise ValueError(f"Unknown model {name}")
