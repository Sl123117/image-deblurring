from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import nn
from torch.nn import functional as F


def compute_psnr(predictions: torch.Tensor, targets: torch.Tensor, max_value: float = 1.0) -> torch.Tensor:
    predictions = predictions.float()
    targets = targets.float()
    mse = torch.mean((predictions - targets) ** 2, dim=(1, 2, 3)).clamp_min(1e-12)
    return 10.0 * torch.log10((max_value ** 2) / mse)


def _gaussian_kernel(window_size: int = 11, sigma: float = 1.5, channels: int = 3) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel_2d = kernel_2d / kernel_2d.sum()
    kernel = kernel_2d.expand(channels, 1, window_size, window_size).contiguous()
    return kernel


def compute_ssim(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    max_value: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    predictions = predictions.float()
    targets = targets.float()
    channels = predictions.size(1)
    kernel = _gaussian_kernel(window_size=window_size, sigma=sigma, channels=channels).to(
        device=predictions.device,
        dtype=predictions.dtype,
    )
    padding = window_size // 2

    mu_x = F.conv2d(predictions, kernel, padding=padding, groups=channels)
    mu_y = F.conv2d(targets, kernel, padding=padding, groups=channels)

    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(predictions * predictions, kernel, padding=padding, groups=channels) - mu_x_sq
    sigma_y_sq = F.conv2d(targets * targets, kernel, padding=padding, groups=channels) - mu_y_sq
    sigma_xy = F.conv2d(predictions * targets, kernel, padding=padding, groups=channels) - mu_xy

    c1 = (0.01 * max_value) ** 2
    c2 = (0.03 * max_value) ** 2

    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    ssim_map = numerator / denominator.clamp_min(1e-12)
    return ssim_map.mean(dim=(1, 2, 3))


@dataclass
class AverageMeter:
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += value * n
        self.count += n

    @property
    def average(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count


def evaluate_model(
    model: nn.Module,
    dataloader,
    *,
    device: torch.device,
    loss_fn: Optional[nn.Module] = None,
    compute_ssim_metric: bool = True,
    max_examples: Optional[int] = None,
) -> dict[str, Any]:
    model.eval()
    loss_meter = AverageMeter()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    per_example: list[dict[str, Any]] = []

    with torch.inference_mode():
        for batch in dataloader:
            inputs = batch["blur"].to(device)
            targets = batch["sharp"].to(device)
            predictions = model(inputs)

            batch_size = inputs.size(0)
            if loss_fn is not None:
                loss_value = float(loss_fn(predictions, targets).item())
                loss_meter.update(loss_value, batch_size)

            batch_psnr = compute_psnr(predictions.detach(), targets)
            batch_psnr_values = [float(value) for value in batch_psnr.detach().cpu().tolist()]
            psnr_meter.update(float(batch_psnr.mean().item()), batch_size)

            batch_ssim_values: list[float] = []
            if compute_ssim_metric:
                batch_ssim = compute_ssim(predictions.detach(), targets)
                batch_ssim_values = [float(value) for value in batch_ssim.detach().cpu().tolist()]
                ssim_meter.update(float(batch_ssim.mean().item()), batch_size)

            metadata = batch.get("metadata")
            if metadata is not None and (max_examples is None or len(per_example) < max_examples):
                remaining = None if max_examples is None else max_examples - len(per_example)
                limit = batch_size if remaining is None else min(batch_size, remaining)
                for index in range(limit):
                    example = {
                        "split": metadata["split"][index],
                        "source_split": metadata["source_split"][index],
                        "sequence": metadata["sequence"][index],
                        "filename": metadata["filename"][index],
                        "blur_path": metadata["blur_path"][index],
                        "sharp_path": metadata["sharp_path"][index],
                        "psnr": batch_psnr_values[index],
                    }
                    if compute_ssim_metric:
                        example["ssim"] = batch_ssim_values[index]
                    per_example.append(example)

    results = {
        "loss": loss_meter.average if loss_meter.count > 0 else None,
        "psnr": psnr_meter.average,
        "num_images": psnr_meter.count,
    }
    if compute_ssim_metric:
        results["ssim"] = ssim_meter.average
    if per_example:
        results["examples"] = per_example
    return results


__all__ = [
    "AverageMeter",
    "compute_psnr",
    "compute_ssim",
    "evaluate_model",
]
