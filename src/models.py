from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F


class IdentityDeblurModel(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs


SUPPORTED_MODEL_NAMES = ("unet", "motion_routed_unet")
MODEL_DISPLAY_NAMES = {
    "unet": "UNetDeblurModel",
    "motion_routed_unet": "MotionRoutedUNet",
}
MODEL_ARTIFACT_STEMS = {
    "unet": "unet_deblurring",
    "motion_routed_unet": "motion_routed_unet_deblurring",
}


def normalize_model_name(model_name: str) -> str:
    normalized = model_name.lower()
    if normalized not in SUPPORTED_MODEL_NAMES:
        raise ValueError(f"Unsupported model_name: {model_name}")
    return normalized


def resolve_model_display_name(model_name: str) -> str:
    return MODEL_DISPLAY_NAMES[normalize_model_name(model_name)]


def resolve_model_artifact_stem(model_name: str) -> str:
    return MODEL_ARTIFACT_STEMS[normalize_model_name(model_name)]


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(inputs))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        upsampled = self.up(inputs)
        diff_y = skip.size(2) - upsampled.size(2)
        diff_x = skip.size(3) - upsampled.size(3)
        if diff_y != 0 or diff_x != 0:
            upsampled = F.pad(
                upsampled,
                [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
            )
        return self.conv(torch.cat([skip, upsampled], dim=1))


class MotionFusionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        hidden_channels = max(out_channels // 4, 8)
        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        )
        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.SiLU(inplace=True),
        )
        self.local_branch = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
        )
        self.context_branch = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=7,
                padding=3,
                groups=out_channels,
                bias=True,
            ),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=7,
                padding=3,
                groups=out_channels,
                bias=True,
            ),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=True),
        )
        self.directional_horizontal = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(1, 9),
            padding=(0, 4),
            groups=out_channels,
            bias=True,
        )
        self.directional_vertical = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(9, 1),
            padding=(4, 0),
            groups=out_channels,
            bias=True,
        )
        self.directional_activation = nn.SiLU(inplace=True)
        self.directional_pointwise = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=True)
        self.directional_output = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.gate_pool = nn.AdaptiveAvgPool2d(1)
        self.gate_reduce = nn.Conv2d(out_channels, hidden_channels, kernel_size=1, bias=True)
        self.gate_activation = nn.SiLU(inplace=True)
        self.gate_expand = nn.Conv2d(hidden_channels, out_channels * 3, kernel_size=1, bias=True)
        self.fusion_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.output_activation = nn.SiLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.residual(inputs)
        shared = self.shared(inputs)

        local_features = self.local_branch(shared)
        context_features = self.context_branch(shared)

        directional_features = self.directional_horizontal(shared) + self.directional_vertical(shared)
        directional_features = self.directional_activation(directional_features)
        directional_features = self.directional_pointwise(directional_features)
        directional_features = self.directional_output(directional_features)

        gate_logits = self.gate_expand(self.gate_activation(self.gate_reduce(self.gate_pool(shared))))
        batch_size, channels = shared.shape[:2]
        gate_weights = gate_logits.view(batch_size, 3, channels, 1, 1).softmax(dim=1)
        stacked_features = torch.stack(
            [local_features, context_features, directional_features],
            dim=1,
        )
        fused = (stacked_features * gate_weights).sum(dim=1)
        fused = self.fusion_conv(fused)
        return self.output_activation(fused + residual)


class MotionDownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.block = MotionFusionBlock(in_channels, out_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(self.pool(inputs))


class MotionUpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.block = MotionFusionBlock(in_channels + skip_channels, out_channels)

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        upsampled = self.up(inputs)
        diff_y = skip.size(2) - upsampled.size(2)
        diff_x = skip.size(3) - upsampled.size(3)
        if diff_y != 0 or diff_x != 0:
            upsampled = F.pad(
                upsampled,
                [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
            )
        return self.block(torch.cat([skip, upsampled], dim=1))


class UNetDeblurModel(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_channels: int = 16):
        super().__init__()
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.input_block = DoubleConv(in_channels, channels[0])
        self.down1 = DownBlock(channels[0], channels[1])
        self.down2 = DownBlock(channels[1], channels[2])
        self.down3 = DownBlock(channels[2], channels[3])
        self.bottleneck = DownBlock(channels[3], channels[3])

        self.up1 = UpBlock(channels[3], channels[3], channels[2])
        self.up2 = UpBlock(channels[2], channels[2], channels[1])
        self.up3 = UpBlock(channels[1], channels[1], channels[0])
        self.up4 = UpBlock(channels[0], channels[0], channels[0])
        self.output_conv = nn.Conv2d(channels[0], out_channels, kernel_size=1)
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        enc1 = self.input_block(inputs)
        enc2 = self.down1(enc1)
        enc3 = self.down2(enc2)
        enc4 = self.down3(enc3)
        bottleneck = self.bottleneck(enc4)

        dec1 = self.up1(bottleneck, enc4)
        dec2 = self.up2(dec1, enc3)
        dec3 = self.up3(dec2, enc2)
        dec4 = self.up4(dec3, enc1)

        residual = self.output_conv(dec4)
        return torch.clamp(inputs + residual, 0.0, 1.0)


class MotionRoutedUNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_channels: int = 16):
        super().__init__()
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.input_block = DoubleConv(in_channels, channels[0])
        self.down1 = DownBlock(channels[0], channels[1])
        self.down2 = MotionDownBlock(channels[1], channels[2])
        self.down3 = MotionDownBlock(channels[2], channels[3])
        self.bottleneck = MotionDownBlock(channels[3], channels[3])

        self.up1 = MotionUpBlock(channels[3], channels[3], channels[2])
        self.up2 = MotionUpBlock(channels[2], channels[2], channels[1])
        self.up3 = UpBlock(channels[1], channels[1], channels[0])
        self.up4 = UpBlock(channels[0], channels[0], channels[0])
        self.output_conv = nn.Conv2d(channels[0], out_channels, kernel_size=1)
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        enc1 = self.input_block(inputs)
        enc2 = self.down1(enc1)
        enc3 = self.down2(enc2)
        enc4 = self.down3(enc3)
        bottleneck = self.bottleneck(enc4)

        dec1 = self.up1(bottleneck, enc4)
        dec2 = self.up2(dec1, enc3)
        dec3 = self.up3(dec2, enc2)
        dec4 = self.up4(dec3, enc1)

        residual = self.output_conv(dec4)
        return torch.clamp(inputs + residual, 0.0, 1.0)


@dataclass(frozen=True)
class ModelSummary:
    model_name: str
    name: str
    trainable_parameters: int
    total_parameters: int
    architecture: str


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return trainable_parameters, total_parameters


def build_deblurring_model(
    *,
    model_name: str,
    base_channels: int,
    in_channels: int = 3,
    out_channels: int = 3,
) -> nn.Module:
    normalized_model_name = normalize_model_name(model_name)
    if normalized_model_name == "unet":
        return UNetDeblurModel(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
        )
    return MotionRoutedUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
    )


def summarize_model(model: nn.Module, name: str, model_name: str | None = None) -> ModelSummary:
    trainable_parameters, total_parameters = count_parameters(model)
    if model_name is not None:
        resolved_model_name = normalize_model_name(model_name)
    else:
        resolved_model_name = next(
            (candidate for candidate, display_name in MODEL_DISPLAY_NAMES.items() if display_name == name),
            None,
        )
        if resolved_model_name is None:
            raise ValueError(
                "summarize_model requires model_name when the display name does not match a supported model."
            )
    return ModelSummary(
        model_name=resolved_model_name,
        name=name,
        trainable_parameters=trainable_parameters,
        total_parameters=total_parameters,
        architecture=str(model),
    )


__all__ = [
    "MODEL_ARTIFACT_STEMS",
    "MODEL_DISPLAY_NAMES",
    "SUPPORTED_MODEL_NAMES",
    "MotionRoutedUNet",
    "IdentityDeblurModel",
    "ModelSummary",
    "UNetDeblurModel",
    "build_deblurring_model",
    "count_parameters",
    "normalize_model_name",
    "resolve_model_artifact_stem",
    "resolve_model_display_name",
    "summarize_model",
]
