from __future__ import annotations

import json
import os
import random
import shutil
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler, LinearLR, SequentialLR, StepLR

from .data import GoProPairDataset, build_dataloader, build_sequence_split
from .metrics import AverageMeter, compute_psnr, evaluate_model
from .models import (
    SUPPORTED_MODEL_NAMES,
    IdentityDeblurModel,
    build_deblurring_model,
    normalize_model_name,
    resolve_model_artifact_stem,
    summarize_model,
)
from .utils import find_dataset_root


@dataclass(frozen=True)
class BenchmarkConfig:
    dataset_root: str = "data/raw/GOPRO_Large"
    seed: int = 42
    val_fraction: float = 0.10
    patch_size: int = 256
    train_batch_size: int = 8
    eval_batch_size: int = 2
    num_workers: int = 4
    cpu_num_threads: int = 4
    num_epochs: int = 2
    learning_rate: float = 2e-4
    loss_name: str = "charbonnier"
    charbonnier_epsilon: float = 1e-3
    scheduler_type: str = "warmup_cosine"
    warmup_epochs: int = 3
    warmup_start_factor: float = 0.1
    cosine_eta_min: float = 1e-6
    scheduler_step_size: int = 15
    scheduler_gamma: float = 0.5
    model_name: str = "unet"
    base_channels: int = 16
    use_amp: bool = True
    validate_each_epoch: bool = True
    resume_checkpoint: Optional[str] = None
    checkpoint_monitor: str = "val_psnr"
    checkpoint_name: Optional[str] = None
    best_checkpoint_name: Optional[str] = None
    latest_checkpoint_name: Optional[str] = None
    artifact_prefix: Optional[str] = None
    pin_memory: Optional[bool] = None
    persistent_workers: bool = True
    prefetch_factor: int = 2
    timing_enabled: bool = True
    report_name: str = "benchmark_report.md"


LEGACY_LOSS_NAME = "l1"
LEGACY_SCHEDULER_TYPE = "step"
SUPPORTED_LOSS_NAMES = {LEGACY_LOSS_NAME, "charbonnier"}
SUPPORTED_SCHEDULER_TYPES = {LEGACY_SCHEDULER_TYPE, "warmup_cosine"}
RECIPE_CONFIG_FIELDS = (
    "loss_name",
    "charbonnier_epsilon",
    "scheduler_type",
    "warmup_epochs",
    "warmup_start_factor",
    "cosine_eta_min",
    "scheduler_step_size",
    "scheduler_gamma",
)


class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon: float = 1e-3):
        super().__init__()
        self.epsilon = float(epsilon)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        difference = predictions - targets
        return torch.sqrt(difference.square() + self.epsilon**2).mean()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _ensure_output_dirs(project_root: Path) -> dict[str, Path]:
    results_root = project_root / "results"
    directories = {
        "results": results_root,
        "figures": results_root / "figures",
        "metrics": results_root / "metrics",
        "checkpoints": results_root / "checkpoints",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def _to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _to_serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_serializable(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_to_serializable(payload), indent=2), encoding="utf-8")


def load_rgb_image(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return (array * 255.0).round().astype(np.uint8)


def resolve_runtime_settings(config: BenchmarkConfig, device: torch.device) -> dict[str, Any]:
    available_cpus = os.cpu_count() or 0
    resolved_num_workers = max(0, config.num_workers)
    if available_cpus:
        resolved_num_workers = min(resolved_num_workers, available_cpus)

    resolved_pin_memory = device.type == "cuda" if config.pin_memory is None else config.pin_memory
    resolved_persistent_workers = bool(config.persistent_workers and resolved_num_workers > 0)
    resolved_prefetch_factor = config.prefetch_factor if resolved_num_workers > 0 else None
    amp_enabled = bool(config.use_amp and device.type == "cuda")

    return {
        "device_type": device.type,
        "available_cpus": available_cpus,
        "amp_enabled": amp_enabled,
        "num_workers": resolved_num_workers,
        "pin_memory": resolved_pin_memory,
        "persistent_workers": resolved_persistent_workers,
        "prefetch_factor": resolved_prefetch_factor,
        "timing_enabled": config.timing_enabled,
    }


def _normalize_loss_name(name: str) -> str:
    normalized = name.lower()
    if normalized not in SUPPORTED_LOSS_NAMES:
        raise ValueError(f"Unsupported loss_name: {name}")
    return normalized


def _normalize_scheduler_type(name: str) -> str:
    normalized = name.lower()
    if normalized not in SUPPORTED_SCHEDULER_TYPES:
        raise ValueError(f"Unsupported scheduler_type: {name}")
    return normalized


def _normalize_recipe_mapping(values: Optional[dict[str, Any]]) -> dict[str, Any]:
    values = dict(values or {})
    loss_name = _normalize_loss_name(str(values.get("loss_name", LEGACY_LOSS_NAME)))
    scheduler_type = _normalize_scheduler_type(str(values.get("scheduler_type", LEGACY_SCHEDULER_TYPE)))

    return {
        "loss_name": loss_name,
        "charbonnier_epsilon": float(values.get("charbonnier_epsilon", BenchmarkConfig.charbonnier_epsilon)),
        "scheduler_type": scheduler_type,
        "warmup_epochs": int(values.get("warmup_epochs", 0 if scheduler_type == LEGACY_SCHEDULER_TYPE else BenchmarkConfig.warmup_epochs)),
        "warmup_start_factor": float(values.get("warmup_start_factor", BenchmarkConfig.warmup_start_factor)),
        "cosine_eta_min": float(values.get("cosine_eta_min", BenchmarkConfig.cosine_eta_min)),
        "scheduler_step_size": int(values.get("scheduler_step_size", BenchmarkConfig.scheduler_step_size)),
        "scheduler_gamma": float(values.get("scheduler_gamma", BenchmarkConfig.scheduler_gamma)),
    }


def _current_recipe_mapping(config: BenchmarkConfig) -> dict[str, Any]:
    values = {field_name: getattr(config, field_name) for field_name in RECIPE_CONFIG_FIELDS}
    return _normalize_recipe_mapping(values)


def _effective_checkpoint_recipe(checkpoint_config: Optional[dict[str, Any]]) -> dict[str, Any]:
    return _normalize_recipe_mapping(checkpoint_config)


def _loss_description(config: BenchmarkConfig) -> str:
    if _normalize_loss_name(config.loss_name) == LEGACY_LOSS_NAME:
        return "L1"
    return f"Charbonnier(eps={config.charbonnier_epsilon:g})"


def _scheduler_description(config: BenchmarkConfig) -> str:
    scheduler_type = _normalize_scheduler_type(config.scheduler_type)
    if scheduler_type == LEGACY_SCHEDULER_TYPE:
        return f"StepLR(step_size={config.scheduler_step_size}, gamma={config.scheduler_gamma})"
    cosine_t_max = config.num_epochs - config.warmup_epochs
    if config.warmup_epochs == 0:
        return f"CosineAnnealingLR(T_max={config.num_epochs}, eta_min={config.cosine_eta_min:g})"
    return (
        "SequentialLR("
        f"LinearLR(start_factor={config.warmup_start_factor:g}, total_iters={config.warmup_epochs}), "
        f"CosineAnnealingLR(T_max={cosine_t_max}, eta_min={config.cosine_eta_min:g}), "
        f"milestones=[{config.warmup_epochs}])"
    )


def build_loss_fn(config: BenchmarkConfig) -> nn.Module:
    if _normalize_loss_name(config.loss_name) == LEGACY_LOSS_NAME:
        return nn.L1Loss()
    return CharbonnierLoss(epsilon=config.charbonnier_epsilon)


def build_scheduler(optimizer: torch.optim.Optimizer, config: BenchmarkConfig) -> LRScheduler:
    scheduler_type = _normalize_scheduler_type(config.scheduler_type)
    if scheduler_type == LEGACY_SCHEDULER_TYPE:
        return StepLR(optimizer, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma)
    if config.warmup_epochs == 0:
        return CosineAnnealingLR(optimizer, T_max=config.num_epochs, eta_min=config.cosine_eta_min)
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=config.warmup_start_factor,
        total_iters=config.warmup_epochs,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.num_epochs - config.warmup_epochs,
        eta_min=config.cosine_eta_min,
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[config.warmup_epochs],
    )


def _validate_config(config: BenchmarkConfig) -> None:
    normalize_model_name(config.model_name)
    if config.num_epochs < 1:
        raise ValueError("num_epochs must be at least 1")
    if config.train_batch_size < 1:
        raise ValueError("train_batch_size must be at least 1")
    if config.eval_batch_size < 1:
        raise ValueError("eval_batch_size must be at least 1")
    if config.num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if config.prefetch_factor < 1:
        raise ValueError("prefetch_factor must be at least 1")
    loss_name = _normalize_loss_name(config.loss_name)
    scheduler_type = _normalize_scheduler_type(config.scheduler_type)
    if loss_name == "charbonnier" and config.charbonnier_epsilon <= 0:
        raise ValueError("charbonnier_epsilon must be greater than 0 when using Charbonnier loss")
    if scheduler_type == LEGACY_SCHEDULER_TYPE:
        if config.scheduler_step_size < 1:
            raise ValueError("scheduler_step_size must be at least 1 for StepLR")
        if config.scheduler_gamma <= 0:
            raise ValueError("scheduler_gamma must be greater than 0 for StepLR")
    else:
        if config.warmup_epochs < 0:
            raise ValueError("warmup_epochs must be non-negative for warmup_cosine scheduler")
        if config.warmup_epochs >= config.num_epochs:
            raise ValueError("warmup_epochs must be less than num_epochs for warmup_cosine scheduler")
        if config.warmup_start_factor <= 0 or config.warmup_start_factor > 1:
            raise ValueError("warmup_start_factor must be in the interval (0, 1]")
        if config.cosine_eta_min < 0:
            raise ValueError("cosine_eta_min must be non-negative")
    if config.checkpoint_monitor not in {"train_loss", "train_psnr", "val_loss", "val_psnr"}:
        raise ValueError(
            "checkpoint_monitor must be one of {'train_loss', 'train_psnr', 'val_loss', 'val_psnr'}"
        )
    if config.checkpoint_monitor.startswith("val_") and not config.validate_each_epoch:
        raise ValueError("validate_each_epoch must be True when checkpoint_monitor uses validation metrics")


def _monitor_mode(name: str) -> str:
    if name.endswith(("psnr", "ssim")):
        return "max"
    if name.endswith("loss"):
        return "min"
    raise ValueError(f"Unsupported checkpoint monitor: {name}")


def _initial_best_metric(name: str) -> float:
    return float("-inf") if _monitor_mode(name) == "max" else float("inf")


def _is_better_metric(current_value: float, best_value: float, monitor_name: str) -> bool:
    if _monitor_mode(monitor_name) == "max":
        return current_value > best_value
    return current_value < best_value


def _resolve_checkpoint_value(epoch_record: dict[str, Any], monitor_name: str) -> float:
    value = epoch_record.get(monitor_name)
    if value is None:
        raise ValueError(f"Checkpoint monitor '{monitor_name}' is unavailable for this epoch.")
    return float(value)


def _resolve_best_checkpoint_name(config: BenchmarkConfig) -> str:
    return config.checkpoint_name or config.best_checkpoint_name or _default_checkpoint_name(config, "best")


def _resolve_latest_checkpoint_name(config: BenchmarkConfig) -> str:
    return config.latest_checkpoint_name or _default_checkpoint_name(config, "latest")


def _default_checkpoint_name(config: BenchmarkConfig, checkpoint_kind: str) -> str:
    model_stem = resolve_model_artifact_stem(config.model_name)
    return f"{model_stem}_{checkpoint_kind}.pt"


def _artifact_filename(config: BenchmarkConfig, filename: str) -> str:
    if not config.artifact_prefix:
        return filename
    return f"{config.artifact_prefix}_{filename}"


def _resolve_resume_checkpoint_path(
    resume_checkpoint: str | Path,
    *,
    project_root: Path,
    output_dirs: dict[str, Path],
) -> Path:
    requested_path = Path(resume_checkpoint)
    if requested_path.is_absolute():
        candidate_paths = [requested_path]
    else:
        candidate_paths = [
            project_root / requested_path,
            output_dirs["checkpoints"] / requested_path,
        ]

    for candidate_path in candidate_paths:
        if candidate_path.exists():
            return candidate_path

    formatted_candidates = ", ".join(str(path) for path in candidate_paths)
    raise FileNotFoundError(
        f"Could not find resume checkpoint '{resume_checkpoint}'. Checked: {formatted_candidates}"
    )


def _known_checkpoint_suffix_pairs() -> tuple[tuple[str, str], ...]:
    pairs = []
    for model_name in SUPPORTED_MODEL_NAMES:
        model_stem = resolve_model_artifact_stem(model_name)
        pairs.append((f"_{model_stem}_latest.pt", f"_{model_stem}_best.pt"))
    return tuple(pairs)


def _load_history_from_metrics_artifact(checkpoint_path: Path) -> list[dict[str, Any]]:
    checkpoint_name = checkpoint_path.name
    artifact_prefix = None
    for latest_suffix, best_suffix in _known_checkpoint_suffix_pairs():
        for suffix in (latest_suffix, best_suffix):
            if checkpoint_name.endswith(suffix):
                artifact_prefix = checkpoint_name.removesuffix(suffix)
                break
        if artifact_prefix is not None:
            break

    if artifact_prefix is None:
        return []

    metrics_path = checkpoint_path.parent.parent / "metrics" / f"{artifact_prefix}_metrics.json"
    if not metrics_path.exists():
        return []

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    history = payload.get("training_history")
    if not isinstance(history, list):
        return []
    return history


def _resolve_resume_best_checkpoint_path(resume_checkpoint_path: Path) -> Optional[Path]:
    checkpoint_name = resume_checkpoint_path.name
    for latest_suffix, best_suffix in _known_checkpoint_suffix_pairs():
        if checkpoint_name.endswith(best_suffix):
            return resume_checkpoint_path if resume_checkpoint_path.exists() else None
        if checkpoint_name.endswith(latest_suffix):
            candidate = resume_checkpoint_path.with_name(
                checkpoint_name.removesuffix(latest_suffix) + best_suffix
            )
            if candidate.exists():
                return candidate
    return None


def _checkpoint_recipe_defaults(checkpoint_path: Path) -> Optional[dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_config = checkpoint.get("config")
    if not isinstance(checkpoint_config, dict):
        return None
    return _effective_checkpoint_recipe(checkpoint_config)


def _maybe_inherit_resume_recipe(config: BenchmarkConfig, resume_checkpoint_path: Path) -> BenchmarkConfig:
    checkpoint_recipe = _checkpoint_recipe_defaults(resume_checkpoint_path)
    if checkpoint_recipe is None:
        return config

    current_recipe = _current_recipe_mapping(config)
    default_recipe = _current_recipe_mapping(BenchmarkConfig())
    if current_recipe != default_recipe:
        return config

    if checkpoint_recipe == current_recipe:
        return config
    return replace(config, **checkpoint_recipe)


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _validate_resume_compatibility(
    checkpoint_config: Optional[dict[str, Any]],
    config: BenchmarkConfig,
) -> None:
    if not checkpoint_config:
        return

    checkpoint_recipe = _effective_checkpoint_recipe(checkpoint_config)
    current_recipe = _current_recipe_mapping(config)
    incompatible_fields = []
    checkpoint_model_name = normalize_model_name(str(checkpoint_config.get("model_name", "unet")))
    current_model_name = normalize_model_name(config.model_name)
    if checkpoint_model_name != current_model_name:
        incompatible_fields.append(("model_name", checkpoint_model_name, current_model_name))
    for field_name in (
        "base_channels",
        "patch_size",
        "seed",
        "val_fraction",
        "learning_rate",
        "checkpoint_monitor",
    ):
        checkpoint_value = checkpoint_config.get(field_name)
        current_value = getattr(config, field_name)
        if checkpoint_value is not None and checkpoint_value != current_value:
            incompatible_fields.append((field_name, checkpoint_value, current_value))

    if checkpoint_recipe["loss_name"] != current_recipe["loss_name"]:
        incompatible_fields.append(("loss_name", checkpoint_recipe["loss_name"], current_recipe["loss_name"]))
    if checkpoint_recipe["loss_name"] == "charbonnier":
        if checkpoint_recipe["charbonnier_epsilon"] != current_recipe["charbonnier_epsilon"]:
            incompatible_fields.append(
                (
                    "charbonnier_epsilon",
                    checkpoint_recipe["charbonnier_epsilon"],
                    current_recipe["charbonnier_epsilon"],
                )
            )

    if checkpoint_recipe["scheduler_type"] != current_recipe["scheduler_type"]:
        incompatible_fields.append(
            ("scheduler_type", checkpoint_recipe["scheduler_type"], current_recipe["scheduler_type"])
        )
    elif checkpoint_recipe["scheduler_type"] == LEGACY_SCHEDULER_TYPE:
        for field_name in ("scheduler_step_size", "scheduler_gamma"):
            if checkpoint_recipe[field_name] != current_recipe[field_name]:
                incompatible_fields.append((field_name, checkpoint_recipe[field_name], current_recipe[field_name]))
    else:
        for field_name in ("warmup_epochs", "warmup_start_factor", "cosine_eta_min"):
            if checkpoint_recipe[field_name] != current_recipe[field_name]:
                incompatible_fields.append((field_name, checkpoint_recipe[field_name], current_recipe[field_name]))

    if incompatible_fields:
        formatted = ", ".join(
            f"{field_name} checkpoint={checkpoint_value!r} current={current_value!r}"
            for field_name, checkpoint_value, current_value in incompatible_fields
        )
        raise ValueError(
            "Resume checkpoint is incompatible with the current config. "
            f"Mismatched fields: {formatted}"
        )


def _autocast_context(*, device: torch.device, amp_enabled: bool):
    if amp_enabled:
        return torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=True)
    return nullcontext()


def _summarize_timing(
    data_wait_times: list[float],
    step_times: list[float],
    *,
    num_images: int,
) -> dict[str, Any]:
    total_data_wait = float(sum(data_wait_times))
    total_step = float(sum(step_times))
    total_time = total_data_wait + total_step
    return {
        "enabled": True,
        "num_steps": len(step_times),
        "num_images": num_images,
        "total_data_wait_seconds": total_data_wait,
        "total_step_seconds": total_step,
        "median_data_wait_seconds": float(np.median(data_wait_times)) if data_wait_times else 0.0,
        "median_step_seconds": float(np.median(step_times)) if step_times else 0.0,
        "images_per_second": float(num_images / total_time) if total_time > 0 else None,
    }


def summarize_training_timing(history: list[dict[str, Any]]) -> dict[str, Any]:
    timing_records = [epoch["timing"] for epoch in history if epoch.get("timing") and epoch["timing"].get("enabled")]
    if not timing_records:
        return {"enabled": False}

    total_images = sum(int(record["num_images"]) for record in timing_records)
    total_data_wait = sum(float(record["total_data_wait_seconds"]) for record in timing_records)
    total_step = sum(float(record["total_step_seconds"]) for record in timing_records)
    total_time = total_data_wait + total_step
    epoch_images_per_second = [record["images_per_second"] for record in timing_records if record["images_per_second"] is not None]

    return {
        "enabled": True,
        "epochs": len(timing_records),
        "total_images": total_images,
        "total_data_wait_seconds": total_data_wait,
        "total_step_seconds": total_step,
        "overall_images_per_second": float(total_images / total_time) if total_time > 0 else None,
        "mean_epoch_images_per_second": float(np.mean(epoch_images_per_second)) if epoch_images_per_second else None,
        "mean_median_data_wait_seconds": float(np.mean([record["median_data_wait_seconds"] for record in timing_records])),
        "mean_median_step_seconds": float(np.mean([record["median_step_seconds"] for record in timing_records])),
    }


def train_one_epoch(
    model: nn.Module,
    dataloader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    amp_enabled: bool,
    scaler: torch.amp.GradScaler,
    timing_enabled: bool,
) -> dict[str, Any]:
    model.train()
    loss_meter = AverageMeter()
    psnr_meter = AverageMeter()
    data_wait_times: list[float] = []
    step_times: list[float] = []
    num_images = 0

    iterator = iter(dataloader)
    while True:
        fetch_start = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            break
        if timing_enabled:
            data_wait_times.append(time.perf_counter() - fetch_start)

        step_start = time.perf_counter()
        inputs = batch["blur"].to(device, non_blocking=amp_enabled)
        targets = batch["sharp"].to(device, non_blocking=amp_enabled)

        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device=device, amp_enabled=amp_enabled):
            predictions = model(inputs)
            loss = loss_fn(predictions, targets)

        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = inputs.size(0)
        num_images += batch_size
        loss_meter.update(float(loss.item()), batch_size)
        batch_psnr = compute_psnr(predictions.detach(), targets)
        psnr_meter.update(float(batch_psnr.mean().item()), batch_size)
        if timing_enabled:
            step_times.append(time.perf_counter() - step_start)

    results: dict[str, Any] = {
        "loss": loss_meter.average,
        "psnr": psnr_meter.average,
    }
    if timing_enabled:
        results["timing"] = _summarize_timing(data_wait_times, step_times, num_images=num_images)
    else:
        results["timing"] = {"enabled": False}
    return results


def _metric_value(metrics: dict[str, Any], name: str) -> str:
    value = metrics.get(name)
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_metric: float,
    monitor_name: str,
    monitor_value: float,
    config: BenchmarkConfig,
    device: torch.device,
    epoch_metrics: Optional[dict[str, Any]] = None,
    training_history: Optional[list[dict[str, Any]]] = None,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "best_metric": best_metric,
            "monitor_name": monitor_name,
            "monitor_value": monitor_value,
            "device_type": device.type,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "config": asdict(config),
            "epoch_metrics": _to_serializable(epoch_metrics) if epoch_metrics is not None else None,
            "training_history": _to_serializable(training_history) if training_history is not None else None,
        },
        path,
    )


def load_training_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LRScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    config: BenchmarkConfig,
    runtime_settings: dict[str, Any],
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device)

    checkpoint_config = checkpoint.get("config")
    if isinstance(checkpoint_config, dict):
        _validate_resume_compatibility(checkpoint_config, config)

    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as exc:
        raise ValueError(
            f"Failed to load model state from resume checkpoint '{path}'. "
            "The checkpoint is not compatible with the current model definition."
        ) from exc

    if "optimizer_state_dict" not in checkpoint or "scheduler_state_dict" not in checkpoint:
        raise ValueError(
            f"Checkpoint '{path}' does not include optimizer and scheduler state required for resume."
        )

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    _move_optimizer_state_to_device(optimizer, device)
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    scaler_state_loaded = False
    scaler_state_dict = checkpoint.get("scaler_state_dict")
    if runtime_settings["amp_enabled"] and scaler_state_dict:
        scaler.load_state_dict(scaler_state_dict)
        scaler_state_loaded = True

    training_history = checkpoint.get("training_history")
    history_source = "checkpoint"
    if not isinstance(training_history, list):
        training_history = _load_history_from_metrics_artifact(path)
        history_source = "metrics" if training_history else "none"

    start_epoch = int(checkpoint.get("epoch", 0))
    if isinstance(training_history, list):
        training_history = training_history[:start_epoch]
    else:
        training_history = []

    return {
        "checkpoint": checkpoint,
        "checkpoint_path": path,
        "start_epoch": start_epoch,
        "best_metric": float(checkpoint.get("best_metric", _initial_best_metric(config.checkpoint_monitor))),
        "monitor_name": checkpoint.get("monitor_name"),
        "monitor_value": checkpoint.get("monitor_value"),
        "training_history": training_history,
        "history_source": history_source,
        "scaler_state_loaded": scaler_state_loaded,
    }


def run_training(
    model: nn.Module,
    train_loader,
    val_loader,
    *,
    device: torch.device,
    config: BenchmarkConfig,
    best_checkpoint_path: Path,
    latest_checkpoint_path: Path,
    runtime_settings: dict[str, Any],
    resume_checkpoint_path: Optional[Path] = None,
) -> tuple[list[dict[str, Any]], float, dict[str, Any]]:
    _validate_config(config)
    loss_fn = build_loss_fn(config)
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    scheduler = build_scheduler(optimizer, config)
    scaler = torch.amp.GradScaler(device="cuda", enabled=runtime_settings["amp_enabled"])

    history: list[dict[str, Any]] = []
    monitor_name = config.checkpoint_monitor
    best_metric = _initial_best_metric(monitor_name)
    start_epoch = 0
    resume_info = {"enabled": False}

    if resume_checkpoint_path is not None:
        resume_state = load_training_checkpoint(
            resume_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            config=config,
            runtime_settings=runtime_settings,
        )
        checkpoint_monitor_name = resume_state["monitor_name"]
        if checkpoint_monitor_name is not None and checkpoint_monitor_name != monitor_name:
            raise ValueError(
                f"Resume checkpoint monitor '{checkpoint_monitor_name}' does not match current monitor '{monitor_name}'."
            )

        start_epoch = resume_state["start_epoch"]
        if config.num_epochs <= start_epoch:
            raise ValueError(
                f"num_epochs={config.num_epochs} must be greater than resume checkpoint epoch {start_epoch}."
            )

        history = list(resume_state["training_history"])
        best_metric = resume_state["best_metric"]

        source_best_checkpoint_path = _resolve_resume_best_checkpoint_path(resume_checkpoint_path)
        if not best_checkpoint_path.exists():
            if source_best_checkpoint_path is not None:
                if source_best_checkpoint_path.resolve() != best_checkpoint_path.resolve():
                    shutil.copy2(source_best_checkpoint_path, best_checkpoint_path)
            else:
                monitor_value = resume_state["monitor_value"]
                if monitor_value is None or float(monitor_value) != best_metric:
                    raise FileNotFoundError(
                        "Could not locate the source best checkpoint for resume. "
                        "Keep the sibling best checkpoint alongside the latest checkpoint when resuming."
                    )
                if resume_checkpoint_path.resolve() != best_checkpoint_path.resolve():
                    shutil.copy2(resume_checkpoint_path, best_checkpoint_path)

        resume_info = {
            "enabled": True,
            "checkpoint_path": str(resume_checkpoint_path),
            "source_best_checkpoint_path": None
            if source_best_checkpoint_path is None
            else str(source_best_checkpoint_path),
            "start_epoch": start_epoch,
            "history_epochs_loaded": len(history),
            "history_source": resume_state["history_source"],
            "scaler_state_loaded": resume_state["scaler_state_loaded"],
            "best_metric_at_resume": best_metric,
        }
        print(
            "Resuming benchmark from {path} at epoch {epoch}.".format(
                path=resume_info["checkpoint_path"],
                epoch=start_epoch,
            ),
            flush=True,
        )

    for epoch in range(start_epoch + 1, config.num_epochs + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]
        train_metrics = train_one_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            loss_fn=loss_fn,
            amp_enabled=runtime_settings["amp_enabled"],
            scaler=scaler,
            timing_enabled=config.timing_enabled,
        )
        val_metrics = None
        if config.validate_each_epoch:
            val_metrics = evaluate_model(
                model,
                val_loader,
                device=device,
                loss_fn=loss_fn,
                compute_ssim_metric=False,
            )
        scheduler.step()

        epoch_record = {
            "epoch": epoch,
            "learning_rate": current_lr,
            "train_loss": train_metrics["loss"],
            "train_psnr": train_metrics["psnr"],
            "val_loss": None if val_metrics is None else val_metrics["loss"],
            "val_psnr": None if val_metrics is None else val_metrics["psnr"],
            "epoch_seconds": time.time() - epoch_start,
            "timing": train_metrics["timing"],
        }
        history.append(epoch_record)

        current_metric = _resolve_checkpoint_value(epoch_record, monitor_name)
        improved = _is_better_metric(current_metric, best_metric, monitor_name)
        checkpoint_best_metric = current_metric if improved else best_metric
        save_checkpoint(
            latest_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_metric=checkpoint_best_metric,
            monitor_name=monitor_name,
            monitor_value=current_metric,
            config=config,
            device=device,
            epoch_metrics=epoch_record,
            training_history=history,
        )

        if improved:
            best_metric = current_metric
            save_checkpoint(
                best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_metric=best_metric,
                monitor_name=monitor_name,
                monitor_value=current_metric,
                config=config,
                device=device,
                epoch_metrics=epoch_record,
                training_history=history,
            )

        message = "epoch={epoch} lr={lr:.6f} train_loss={train_loss:.4f} train_psnr={train_psnr:.3f}".format(
            epoch=epoch_record["epoch"],
            lr=epoch_record["learning_rate"],
            train_loss=epoch_record["train_loss"],
            train_psnr=epoch_record["train_psnr"],
        )
        if val_metrics is not None:
            message += " val_loss={val_loss:.4f} val_psnr={val_psnr:.3f}".format(
                epoch=epoch_record["epoch"],
                val_loss=epoch_record["val_loss"],
                val_psnr=epoch_record["val_psnr"],
            )
        timing = epoch_record["timing"]
        if timing.get("enabled") and timing.get("images_per_second") is not None:
            message += " ips={images_per_second:.2f} data_wait={data_wait:.4f}s step={step:.4f}s".format(
                images_per_second=timing["images_per_second"],
                data_wait=timing["median_data_wait_seconds"],
                step=timing["median_step_seconds"],
            )
        message += " time={time_seconds:.1f}s".format(time_seconds=epoch_record["epoch_seconds"])
        print(message, flush=True)

    return history, best_metric, resume_info


def load_checkpoint(path: Path, model: nn.Module, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def save_comparison_figure(
    *,
    split_name: str,
    example: dict[str, Any],
    model: nn.Module,
    model_label: str,
    identity_model: nn.Module,
    device: torch.device,
    output_path: Path,
) -> dict[str, float]:
    blur_image = load_rgb_image(example["blur_path"])
    sharp_image = load_rgb_image(example["sharp_path"])

    blur_tensor = torch.from_numpy(blur_image.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    sharp_tensor = torch.from_numpy(sharp_image.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.inference_mode():
        identity_prediction = identity_model(blur_tensor)
        model_prediction = model(blur_tensor)

    identity_psnr = float(compute_psnr(identity_prediction, sharp_tensor).item())
    model_psnr = float(compute_psnr(model_prediction, sharp_tensor).item())

    identity_image = tensor_to_image(identity_prediction.squeeze(0))
    model_image = tensor_to_image(model_prediction.squeeze(0))

    figure, axes = plt.subplots(1, 4, figsize=(18, 4.5), constrained_layout=True)
    panels = [
        (blur_image, "Blur input"),
        (identity_image, f"Identity\nPSNR {identity_psnr:.2f}"),
        (model_image, f"{model_label}\nPSNR {model_psnr:.2f}"),
        (sharp_image, "Sharp target"),
    ]

    for axis, (image, title) in zip(axes, panels):
        axis.imshow(image)
        axis.set_title(title)
        axis.axis("off")

    figure.suptitle(f"{split_name}: {example['sequence']} / {example['filename']}")
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return {
        "identity_psnr": identity_psnr,
        "model_psnr": model_psnr,
    }


def write_benchmark_report(
    path: Path,
    *,
    config: BenchmarkConfig,
    device: torch.device,
    runtime_settings: dict[str, Any],
    resume_info: Optional[dict[str, Any]],
    split_manifest: dict[str, Any],
    model_summary: dict[str, Any],
    identity_metrics: dict[str, Any],
    model_metrics: dict[str, Any],
    training_history: list[dict[str, Any]],
) -> None:
    train_sequences = ", ".join(split_manifest["train_sequences"])
    val_sequences = ", ".join(split_manifest["val_sequences"])
    val_psnr_gain = model_metrics["val"]["psnr"] - identity_metrics["val"]["psnr"]
    val_ssim_gain = model_metrics["val"]["ssim"] - identity_metrics["val"]["ssim"]
    test_psnr_gain = model_metrics["test"]["psnr"] - identity_metrics["test"]["psnr"]
    test_ssim_gain = model_metrics["test"]["ssim"] - identity_metrics["test"]["ssim"]
    model_label = model_summary["name"]

    qualitative_lines = [
        f"- The identity baseline is already strong on GoPro, so the {model_label} benchmark needs to clear a fairly high floor."
    ]
    if test_psnr_gain < 0.05:
        qualitative_lines.append(
            "- This run only improves very slightly over identity, so the network is learning some residual correction but not enough to materially change the benchmark yet."
        )
    else:
        qualitative_lines.append(
            f"- The {model_label} improves over identity on both validation and test, which confirms that the paired pipeline and restoration model are working end to end."
        )

    issue_lines = [
        f"- This run uses a modest {model_label} baseline, so model capacity and training duration are still conservative.",
        "- Validation is derived from only three sequences, so conclusions should be treated as directional rather than final.",
    ]
    if config.validate_each_epoch:
        issue_lines.append(
            f"- The best checkpoint is selected by `{config.checkpoint_monitor}`, but worker and batch-size settings are still conservative rather than explicitly tuned."
        )
    else:
        issue_lines.append(
            "- The checkpoint was selected without per-epoch validation, so future runs should keep validation-based model selection enabled."
        )
    if test_psnr_gain < 0.05:
        issue_lines.append(
            "- The current setup is too weak to produce a meaningful margin over the identity baseline; before stronger model families, the next fix should be more training budget and validation-based checkpoint selection."
        )
    else:
        issue_lines.append(
            "- Full-resolution evaluation is deterministic, but training still samples one random patch per frame per epoch; longer runs should improve stability."
        )

    lines = [
        "# Benchmark Report",
        "",
    ]
    if config.artifact_prefix:
        lines.extend(
            [
                "## Run",
                "",
                f"- Artifact prefix: `{config.artifact_prefix}`",
                "",
            ]
        )
    if resume_info and resume_info.get("enabled"):
        lines.extend(
            [
                "## Resume",
                "",
                f"- Resumed from checkpoint: `{resume_info['checkpoint_path']}`",
                f"- Resume start epoch: {resume_info['start_epoch']}",
                f"- History restored from: `{resume_info['history_source']}`",
                f"- Restored epochs in history: {resume_info['history_epochs_loaded']}",
                f"- AMP scaler state restored: {resume_info['scaler_state_loaded']}",
                "",
            ]
        )

    lines.extend(
        [
        "## Train/Validation Split",
        "",
        f"- Seed: `{config.seed}`",
        f"- Validation fraction from official training sequences: {config.val_fraction:.0%}",
        f"- Train sequences ({len(split_manifest['train_sequences'])}): {train_sequences}",
        f"- Validation sequences ({len(split_manifest['val_sequences'])}): {val_sequences}",
        "",
        "## Model",
        "",
        f"- Model key: `{model_summary['model_name']}`",
        f"- Architecture: `{model_summary['name']}`",
        f"- Trainable parameters: {model_summary['trainable_parameters']:,}",
        f"- Total parameters: {model_summary['total_parameters']:,}",
        f"- Base channels: {config.base_channels}",
        f"- Device used: `{device.type}`",
        f"- AMP enabled on CUDA: {runtime_settings['amp_enabled']}",
        "",
        "## Training Setup",
        "",
        f"- Patch size: {config.patch_size}x{config.patch_size}",
        f"- Batch size: {config.train_batch_size}",
        f"- Eval batch size: {config.eval_batch_size}",
        f"- DataLoader workers: {runtime_settings['num_workers']}",
        f"- Pin memory: {runtime_settings['pin_memory']}",
        f"- Persistent workers: {runtime_settings['persistent_workers']}",
        f"- Prefetch factor: {runtime_settings['prefetch_factor']}",
        f"- CPU threads: {config.cpu_num_threads}",
        f"- Epochs: {config.num_epochs}",
        f"- Validate each epoch: {config.validate_each_epoch}",
        f"- Checkpoint monitor: {config.checkpoint_monitor}",
        f"- Optimizer: Adam",
        f"- Learning rate: {config.learning_rate}",
        f"- Scheduler: {_scheduler_description(config)}",
        f"- Loss: {_loss_description(config)}",
        "",
        "## Metrics",
        "",
        f"- Identity validation: PSNR {identity_metrics['val']['psnr']:.4f}, SSIM {identity_metrics['val']['ssim']:.4f}",
        f"- Identity test: PSNR {identity_metrics['test']['psnr']:.4f}, SSIM {identity_metrics['test']['ssim']:.4f}",
        f"- {model_label} validation: PSNR {model_metrics['val']['psnr']:.4f}, SSIM {model_metrics['val']['ssim']:.4f}",
        f"- {model_label} test: PSNR {model_metrics['test']['psnr']:.4f}, SSIM {model_metrics['test']['ssim']:.4f}",
        f"- Validation improvement over identity: PSNR {val_psnr_gain:+.4f}, SSIM {val_ssim_gain:+.4f}",
        f"- Test improvement over identity: PSNR {test_psnr_gain:+.4f}, SSIM {test_ssim_gain:+.4f}",
        "",
        "## Training History",
        "",
        ]
    )

    for epoch in training_history:
        line = "- Epoch {epoch}: train_loss {train_loss:.4f}, train_psnr {train_psnr:.4f}".format(**epoch)
        if epoch["val_loss"] is not None and epoch["val_psnr"] is not None:
            line += ", val_loss {val_loss:.4f}, val_psnr {val_psnr:.4f}".format(**epoch)
        if epoch.get("timing", {}).get("enabled") and epoch["timing"].get("images_per_second") is not None:
            line += ", ips {images_per_second:.2f}".format(**epoch["timing"])
        line += ", lr {learning_rate:.6f}, time {epoch_seconds:.1f}s".format(**epoch)
        lines.append(line)

    lines.extend(
        [
            "",
            "## Qualitative Observations",
            "",
            *qualitative_lines,
            "",
            "## Issues Before Stronger Models",
            "",
            *issue_lines,
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_benchmark(config: Optional[BenchmarkConfig] = None) -> dict[str, Any]:
    config = config or BenchmarkConfig()
    project_root = Path(__file__).resolve().parents[1]
    output_dirs = _ensure_output_dirs(project_root)
    resume_checkpoint_path = None
    if config.resume_checkpoint:
        resume_checkpoint_path = _resolve_resume_checkpoint_path(
            config.resume_checkpoint,
            project_root=project_root,
            output_dirs=output_dirs,
        )
        config = _maybe_inherit_resume_recipe(config, resume_checkpoint_path)
    config = replace(config, model_name=normalize_model_name(config.model_name))

    _validate_config(config)
    seed_everything(config.seed)

    dataset_root = find_dataset_root(config.dataset_root)
    device = select_device()
    runtime_settings = resolve_runtime_settings(config, device)
    if device.type == "cpu":
        torch.set_num_threads(config.cpu_num_threads)
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(1)

    split = build_sequence_split(root=dataset_root, val_fraction=config.val_fraction, seed=config.seed)
    split_manifest = {
        "seed": split.seed,
        "val_fraction": split.val_fraction,
        "dataset_root": str(dataset_root),
        "train_sequences": split.train_sequences,
        "val_sequences": split.val_sequences,
        "test_sequences": split.test_sequences,
    }
    write_json(
        output_dirs["metrics"] / _artifact_filename(config, "sequence_split.json"),
        split_manifest,
    )

    train_dataset = GoProPairDataset(
        root=dataset_root,
        split="train",
        mode="train",
        patch_size=config.patch_size,
        val_fraction=config.val_fraction,
        seed=config.seed,
    )
    val_dataset = GoProPairDataset(
        root=dataset_root,
        split="val",
        mode="eval",
        patch_size=None,
        val_fraction=config.val_fraction,
        seed=config.seed,
    )
    test_dataset = GoProPairDataset(
        root=dataset_root,
        split="test",
        mode="eval",
        patch_size=None,
        val_fraction=config.val_fraction,
        seed=config.seed,
    )

    train_loader = build_dataloader(
        train_dataset,
        batch_size=config.train_batch_size,
        num_workers=runtime_settings["num_workers"],
        pin_memory=runtime_settings["pin_memory"],
        persistent_workers=runtime_settings["persistent_workers"],
        prefetch_factor=runtime_settings["prefetch_factor"],
    )
    val_loader = build_dataloader(
        val_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=runtime_settings["num_workers"],
        pin_memory=runtime_settings["pin_memory"],
        persistent_workers=runtime_settings["persistent_workers"],
        prefetch_factor=runtime_settings["prefetch_factor"],
    )
    test_loader = build_dataloader(
        test_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=runtime_settings["num_workers"],
        pin_memory=runtime_settings["pin_memory"],
        persistent_workers=runtime_settings["persistent_workers"],
        prefetch_factor=runtime_settings["prefetch_factor"],
    )

    loss_fn = build_loss_fn(config)
    identity_model = IdentityDeblurModel().to(device)
    model = build_deblurring_model(model_name=config.model_name, base_channels=config.base_channels).to(device)
    summary = summarize_model(model, name=type(model).__name__, model_name=config.model_name)
    write_json(
        output_dirs["metrics"] / _artifact_filename(config, "model_summary.json"),
        asdict(summary),
    )

    print("Evaluating identity baseline on validation split...", flush=True)
    identity_val_metrics = evaluate_model(identity_model, val_loader, device=device, loss_fn=loss_fn, compute_ssim_metric=True)
    print("Evaluating identity baseline on test split...", flush=True)
    identity_test_metrics = evaluate_model(identity_model, test_loader, device=device, loss_fn=loss_fn, compute_ssim_metric=True)

    best_checkpoint_path = output_dirs["checkpoints"] / _resolve_best_checkpoint_name(config)
    latest_checkpoint_path = output_dirs["checkpoints"] / _resolve_latest_checkpoint_name(config)
    print(f"Training {summary.name} benchmark...", flush=True)
    training_history, best_checkpoint_metric, resume_info = run_training(
        model,
        train_loader,
        val_loader,
        device=device,
        config=config,
        best_checkpoint_path=best_checkpoint_path,
        latest_checkpoint_path=latest_checkpoint_path,
        runtime_settings=runtime_settings,
        resume_checkpoint_path=resume_checkpoint_path,
    )

    load_checkpoint(best_checkpoint_path, model, device)
    print(f"Evaluating best {summary.name} on validation split...", flush=True)
    model_val_metrics = evaluate_model(model, val_loader, device=device, loss_fn=loss_fn, compute_ssim_metric=True, max_examples=4)
    print(f"Evaluating best {summary.name} on test split...", flush=True)
    model_test_metrics = evaluate_model(model, test_loader, device=device, loss_fn=loss_fn, compute_ssim_metric=True, max_examples=4)

    comparison_examples = []
    if model_val_metrics.get("examples"):
        comparison_examples.append(("val", model_val_metrics["examples"][0]))
        comparison_examples.append(("val", model_val_metrics["examples"][-1]))
    if model_test_metrics.get("examples"):
        comparison_examples.append(("test", model_test_metrics["examples"][0]))
        comparison_examples.append(("test", model_test_metrics["examples"][-1]))

    visual_summaries = []
    for split_name, example in comparison_examples:
        figure_name = _artifact_filename(
            config,
            f"benchmark_{split_name}_{example['sequence']}_{Path(example['filename']).stem}.png",
        )
        output_path = output_dirs["figures"] / figure_name
        visual_metrics = save_comparison_figure(
            split_name=split_name,
            example=example,
            model=model,
            model_label=summary.name,
            identity_model=identity_model,
            device=device,
            output_path=output_path,
        )
        visual_summaries.append(
            {
                "split": split_name,
                "sequence": example["sequence"],
                "filename": example["filename"],
                "path": str(output_path),
                **visual_metrics,
            }
        )

    metrics_payload = {
        "config": asdict(config),
        "device": device.type,
        "runtime": runtime_settings,
        "resume": resume_info,
        "split_manifest": split_manifest,
        "model_summary": asdict(summary),
        "identity": {
            "val": identity_val_metrics,
            "test": identity_test_metrics,
        },
        "model": {
            "best_checkpoint_metric": best_checkpoint_metric,
            "best_checkpoint_path": str(best_checkpoint_path),
            "latest_checkpoint_path": str(latest_checkpoint_path),
            "val": model_val_metrics,
            "test": model_test_metrics,
        },
        "training_history": training_history,
        "training_timing_summary": summarize_training_timing(training_history),
        "visual_comparisons": visual_summaries,
    }
    if config.model_name == "unet":
        metrics_payload["unet"] = metrics_payload["model"]
    write_json(
        output_dirs["metrics"] / _artifact_filename(config, "benchmark_metrics.json"),
        metrics_payload,
    )
    write_benchmark_report(
        output_dirs["metrics"] / config.report_name,
        config=config,
        device=device,
        runtime_settings=runtime_settings,
        resume_info=resume_info,
        split_manifest=split_manifest,
        model_summary=asdict(summary),
        identity_metrics={"val": identity_val_metrics, "test": identity_test_metrics},
        model_metrics={"val": model_val_metrics, "test": model_test_metrics},
        training_history=training_history,
    )

    print("Identity val/test:", _metric_value(identity_val_metrics, "psnr"), _metric_value(identity_test_metrics, "psnr"), flush=True)
    print(
        f"{summary.name} val/test:",
        _metric_value(model_val_metrics, "psnr"),
        _metric_value(model_test_metrics, "psnr"),
        flush=True,
    )
    print("Best checkpoint:", best_checkpoint_path, flush=True)
    print("Latest checkpoint:", latest_checkpoint_path, flush=True)
    return metrics_payload


if __name__ == "__main__":
    run_benchmark()
