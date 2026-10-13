"""Model factory for HALO."""

from .fusion import AsymBiChaFuse
from .model_ACM import ASKCResUNet
from .model_ALCNet import ASKCResNetFPN
from .model_DNANet import DNANet, Res_CBAM_block
from dataset import load_param


def build_model(name, in_channels=1, deep_supervision=True, channel_size="three", backbone="resnet_18"):
    name = name.lower()
    if name == "dnanet":
        nb_filter, num_blocks = load_param(channel_size, backbone)
        return DNANet(
            num_classes=1,
            input_channels=in_channels,
            block=Res_CBAM_block,
            num_blocks=num_blocks,
            nb_filter=nb_filter,
            deep_supervision=deep_supervision,
        )
    if name == "acm":
        return ASKCResUNet(
            in_channels=in_channels,
            layers=[3, 3, 3],
            channels=[8, 16, 32, 64],
            fuse_mode="AsymBi",
            tiny=False,
            classes=1,
        )
    if name == "alcnet":
        return ASKCResNetFPN(
            in_channels=in_channels,
            layers=[4, 4, 4],
            channels=[8, 16, 32, 64],
            fuse_mode="AsymBi",
            act_dilation=16,
            classes=1,
        )
    if name == "sctransnet":
        # Lazy import: only needs the cloned external repo when this backbone is used.
        from .external_backbones import SCTransNetWrapper
        return SCTransNetWrapper(in_channels=in_channels, deepsuper=deep_supervision)
    if name == "mshnet":
        from .external_backbones import MSHNetWrapper
        return MSHNetWrapper(in_channels=in_channels)
    raise ValueError(f"Unsupported backbone: {name}")


__all__ = [
    "AsymBiChaFuse",
    "ASKCResNetFPN",
    "ASKCResUNet",
    "DNANet",
    "Res_CBAM_block",
    "build_model",
]
