from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F


class IdentityDeblurModel(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs


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


@dataclass(frozen=True)
class ModelSummary:
    name: str
    trainable_parameters: int
    total_parameters: int
    architecture: str


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return trainable_parameters, total_parameters


def summarize_model(model: nn.Module, name: str) -> ModelSummary:
    trainable_parameters, total_parameters = count_parameters(model)
    return ModelSummary(
        name=name,
        trainable_parameters=trainable_parameters,
        total_parameters=total_parameters,
        architecture=str(model),
    )


__all__ = [
    "IdentityDeblurModel",
    "ModelSummary",
    "UNetDeblurModel",
    "count_parameters",
    "summarize_model",
]
