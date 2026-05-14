from __future__ import annotations

import json
import os
import re
from dataclasses import replace
from typing import Any

import modal

APP_NAME = "image-deblurring-benchmark"
PROJECT_ROOT = "/root/project"
DATA_ROOT = f"{PROJECT_ROOT}/data/raw"
RESULTS_ROOT = f"{PROJECT_ROOT}/results"
DATA_VOLUME_NAME = "image-deblurring-gopro-data"
RESULTS_VOLUME_NAME = "image-deblurring-results"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")
    .env(
        {
            "PYTHONPATH": PROJECT_ROOT,
            "PYTHONUNBUFFERED": "1",
            "MPLCONFIGDIR": "/tmp/matplotlib",
        }
    )
    .add_local_dir("src", remote_path=f"{PROJECT_ROOT}/src")
    .add_local_file("modal_app.py", remote_path=f"{PROJECT_ROOT}/modal_app.py")
    .add_local_file("README.md", remote_path=f"{PROJECT_ROOT}/README.md")
)

data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
results_volume = modal.Volume.from_name(RESULTS_VOLUME_NAME, create_if_missing=True)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-").lower()
    return slug or "modal"


def _build_benchmark_config(
    *,
    profile: str,
    run_name: str,
    epochs: int,
    train_batch_size: int,
    eval_batch_size: int,
    num_workers: int,
    base_channels: int,
    model_name: str,
    resume_checkpoint: str,
):
    from src.benchmark import BenchmarkConfig
    from src.models import normalize_model_name, resolve_model_artifact_stem

    normalized_profile = profile.lower()
    normalized_model_name = normalize_model_name(model_name)
    checkpoint_prefix = _slugify(run_name)
    model_artifact_stem = resolve_model_artifact_stem(normalized_model_name)
    report_name = f"{checkpoint_prefix}_{normalized_profile}_benchmark_report.md"
    best_checkpoint_name = f"{checkpoint_prefix}_{normalized_profile}_{model_artifact_stem}_best.pt"
    latest_checkpoint_name = f"{checkpoint_prefix}_{normalized_profile}_{model_artifact_stem}_latest.pt"

    config = BenchmarkConfig(
        dataset_root="data/raw/GOPRO_Large",
        model_name=normalized_model_name,
        artifact_prefix=checkpoint_prefix,
        report_name=report_name,
        best_checkpoint_name=best_checkpoint_name,
        latest_checkpoint_name=latest_checkpoint_name,
        resume_checkpoint=resume_checkpoint or None,
    )

    if normalized_profile == "smoke":
        config = replace(
            config,
            num_epochs=1,
            train_batch_size=2,
            eval_batch_size=1,
            num_workers=2,
            cpu_num_threads=2,
            base_channels=16,
        )
    elif normalized_profile in {"benchmark", "default"}:
        config = replace(
            config,
            num_epochs=60,
            train_batch_size=16,
            eval_batch_size=4,
            num_workers=4,
            cpu_num_threads=4,
            base_channels=32,
        )
    else:
        raise ValueError("profile must be one of {'smoke', 'benchmark', 'default'}")

    overrides: dict[str, int] = {}
    if epochs > 0:
        overrides["num_epochs"] = epochs
    if train_batch_size > 0:
        overrides["train_batch_size"] = train_batch_size
    if eval_batch_size > 0:
        overrides["eval_batch_size"] = eval_batch_size
    if num_workers >= 0:
        overrides["num_workers"] = num_workers
    if base_channels > 0:
        overrides["base_channels"] = base_channels
    if overrides:
        config = replace(config, **overrides)

    return config


def _run_smoke_check(*, model_name: str = "unet", base_channels: int = 16) -> dict[str, Any]:
    import torch
    from torch.optim import Adam
    from torch.utils.data import Subset

    from src.benchmark import (
        BenchmarkConfig,
        build_loss_fn,
        resolve_runtime_settings,
        seed_everything,
        select_device,
        train_one_epoch,
    )
    from src.data import GoProPairDataset, build_dataloader, build_sequence_split
    from src.metrics import evaluate_model
    from src.models import build_deblurring_model, normalize_model_name, summarize_model

    os.chdir(PROJECT_ROOT)
    seed_everything(42)
    normalized_model_name = normalize_model_name(model_name)

    config = BenchmarkConfig(
        dataset_root="data/raw/GOPRO_Large",
        model_name=normalized_model_name,
        num_epochs=1,
        train_batch_size=2,
        eval_batch_size=1,
        num_workers=2,
        cpu_num_threads=2,
        base_channels=base_channels,
        use_amp=True,
        validate_each_epoch=True,
        checkpoint_monitor="val_psnr",
        timing_enabled=True,
    )

    device = select_device()
    runtime = resolve_runtime_settings(config, device)

    split = build_sequence_split(root=config.dataset_root, val_fraction=config.val_fraction, seed=config.seed)
    train_dataset = GoProPairDataset(
        root=config.dataset_root,
        split="train",
        mode="train",
        patch_size=config.patch_size,
        val_fraction=config.val_fraction,
        seed=config.seed,
    )
    val_dataset = GoProPairDataset(
        root=config.dataset_root,
        split="val",
        mode="eval",
        patch_size=None,
        val_fraction=config.val_fraction,
        seed=config.seed,
    )

    train_loader = build_dataloader(
        Subset(train_dataset, list(range(4))),
        batch_size=config.train_batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=runtime["num_workers"],
        pin_memory=runtime["pin_memory"],
        persistent_workers=runtime["persistent_workers"],
        prefetch_factor=runtime["prefetch_factor"],
    )
    val_loader = build_dataloader(
        Subset(val_dataset, list(range(2))),
        batch_size=config.eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=runtime["num_workers"],
        pin_memory=runtime["pin_memory"],
        persistent_workers=runtime["persistent_workers"],
        prefetch_factor=runtime["prefetch_factor"],
    )

    model = build_deblurring_model(model_name=config.model_name, base_channels=config.base_channels).to(device)
    summary = summarize_model(model, name=type(model).__name__, model_name=config.model_name)
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = build_loss_fn(config)
    scaler = torch.amp.GradScaler(device="cuda", enabled=runtime["amp_enabled"])

    train_metrics = train_one_epoch(
        model,
        train_loader,
        device=device,
        optimizer=optimizer,
        loss_fn=loss_fn,
        amp_enabled=runtime["amp_enabled"],
        scaler=scaler,
        timing_enabled=config.timing_enabled,
    )
    val_metrics = evaluate_model(
        model,
        val_loader,
        device=device,
        loss_fn=loss_fn,
        compute_ssim_metric=True,
        max_examples=2,
    )

    return {
        "profile": "smoke",
        "device": device.type,
        "model_name": config.model_name,
        "model_summary": summary.__dict__,
        "runtime": runtime,
        "split_manifest": {
            "seed": split.seed,
            "val_fraction": split.val_fraction,
            "train_sequences": split.train_sequences,
            "val_sequences": split.val_sequences,
            "test_sequences": split.test_sequences,
        },
        "train_pairs_checked": len(train_loader.dataset),
        "val_pairs_checked": len(val_loader.dataset),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
    }


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=16384,
    timeout=60 * 60 * 6,
    startup_timeout=60 * 20,
    include_source=False,
    volumes={
        DATA_ROOT: data_volume,
        RESULTS_ROOT: results_volume,
    },
)
def run_benchmark_remote(
    profile: str = "smoke",
    run_name: str = "modal",
    epochs: int = 0,
    train_batch_size: int = 0,
    eval_batch_size: int = 0,
    num_workers: int = -1,
    base_channels: int = 0,
    model_name: str = "unet",
    resume_checkpoint: str = "",
) -> dict[str, Any]:
    from src.benchmark import run_benchmark

    os.chdir(PROJECT_ROOT)
    os.makedirs(f"{DATA_ROOT}/GOPRO_Large", exist_ok=True)
    os.makedirs(RESULTS_ROOT, exist_ok=True)

    config = _build_benchmark_config(
        profile=profile,
        run_name=run_name,
        epochs=epochs,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
        base_channels=base_channels,
        model_name=model_name,
        resume_checkpoint=resume_checkpoint,
    )

    metrics = run_benchmark(config)
    results_volume.commit()

    return {
        "profile": profile,
        "run_name": run_name,
        "model_name": metrics["config"]["model_name"],
        "device": metrics["device"],
        "runtime": metrics["runtime"],
        "resume": metrics["resume"],
        "report_path": f"{RESULTS_ROOT}/metrics/{config.report_name}",
        "best_checkpoint_path": metrics["model"]["best_checkpoint_path"],
        "latest_checkpoint_path": metrics["model"]["latest_checkpoint_path"],
        "val_psnr": metrics["model"]["val"]["psnr"],
        "test_psnr": metrics["model"]["test"]["psnr"],
    }


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=16384,
    timeout=60 * 30,
    startup_timeout=60 * 20,
    include_source=False,
    volumes={
        DATA_ROOT: data_volume,
        RESULTS_ROOT: results_volume,
    },
)
def run_smoke_remote(model_name: str = "unet", base_channels: int = 16) -> dict[str, Any]:
    os.chdir(PROJECT_ROOT)
    os.makedirs(f"{DATA_ROOT}/GOPRO_Large", exist_ok=True)
    os.makedirs(RESULTS_ROOT, exist_ok=True)

    smoke_result = _run_smoke_check(model_name=model_name, base_channels=base_channels)
    return smoke_result


@app.local_entrypoint()
def main(
    profile: str = "smoke",
    run_name: str = "modal",
    epochs: int = 0,
    train_batch_size: int = 0,
    eval_batch_size: int = 0,
    num_workers: int = -1,
    base_channels: int = 0,
    model_name: str = "unet",
    resume_checkpoint: str = "",
) -> None:
    normalized_profile = profile.lower()
    if normalized_profile == "smoke":
        smoke_base_channels = base_channels if base_channels > 0 else 16
        result = run_smoke_remote.remote(model_name=model_name, base_channels=smoke_base_channels)
    else:
        result = run_benchmark_remote.remote(
            profile=profile,
            run_name=run_name,
            epochs=epochs,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            base_channels=base_channels,
            model_name=model_name,
            resume_checkpoint=resume_checkpoint,
        )
    print(json.dumps(result, indent=2))
