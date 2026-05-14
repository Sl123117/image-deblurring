# Single-Image Motion Deblurring Benchmark

This repository is set up for paired blurry/sharp image inspection and data loading. The current foundation targets the locally available GoPro Large dataset and stops at dataset validation, reusable dataloading, and sample visualization.

## Project Layout

```text
image-deblurring/
├── README.md
├── modal_app.py
├── requirements.txt
├── data/
│   ├── raw/
│   │   └── GOPRO_Large/
│   └── processed/
├── notebooks/
├── results/
│   ├── checkpoints/
│   ├── figures/
│   └── metrics/
└── src/
    ├── __init__.py
    ├── data.py
    ├── transforms.py
    └── utils.py
```

## Dataset Notes

- Canonical dataset path for the codebase: `data/raw/GOPRO_Large`
- Expected sequence layout:

```text
GOPRO_Large/
├── train/
│   └── <sequence_name>/
│       ├── blur/
│       ├── blur_gamma/
│       └── sharp/
└── test/
    └── <sequence_name>/
        ├── blur/
        ├── blur_gamma/
        └── sharp/
```

## Quick Start

```bash
.venv/bin/python -m pip install -r requirements.txt
```

```python
from src.data import GoProPairDataset, build_dataloader

train_dataset = GoProPairDataset(split="train", patch_size=256, val_fraction=0.10, seed=42)
val_dataset = GoProPairDataset(split="val", mode="eval", val_fraction=0.10, seed=42)
test_dataset = GoProPairDataset(split="test", mode="eval")

train_loader = build_dataloader(train_dataset, batch_size=8, num_workers=4)
```

The latest dataset audit is written to `results/metrics/dataset_report.md`, and paired sample figures are saved under `results/figures/`.

## GitHub Tracking

This repository is intended to track source code, documentation, benchmark reports, metrics JSON, and small figure outputs.

The following stay local and are excluded from normal GitHub commits:

- `data/raw/` and `data/processed/` dataset artifacts
- `results/checkpoints/` model checkpoints and resume files
- local environment directories such as `.venv/` and `.pycache_tmp/`

## Modal GPU Runs

This repo includes `modal_app.py` so the existing benchmark can run on a Modal `L4` while keeping the same project-root-relative `data/` and `results/` layout inside the container.

Create persistent Modal volumes:

```bash
.venv/bin/python -m modal volume create image-deblurring-gopro-data
.venv/bin/python -m modal volume create image-deblurring-results
```

Upload the GoPro dataset once:

```bash
.venv/bin/python -m modal volume put image-deblurring-gopro-data data/raw/GOPRO_Large /GOPRO_Large
```

Run a remote smoke test first. This checks the dataset mount, CUDA path, one tiny training pass, and one tiny validation pass without launching the full benchmark:

```bash
.venv/bin/python -m modal run modal_app.py --profile smoke --run-name modal-smoke
```

Run the benchmark wrapper later. The `benchmark` profile now defaults to a first serious L4 configuration:

- `60` epochs
- `base_channels=32`
- `Charbonnier(eps=1e-3)` loss
- `LinearLR` warmup for `3` epochs followed by `CosineAnnealingLR`
- `train_batch_size=16`
- `eval_batch_size=4`
- `num_workers=4`
- `6` hour Modal timeout

All benchmark artifacts are namespaced by `run_name`, so repeated Modal runs do not overwrite each other.

```bash
.venv/bin/python -m modal run modal_app.py --profile benchmark --run-name l4-charb-cosine-20260511-01
```

Override any of those defaults if needed:

```bash
.venv/bin/python -m modal run modal_app.py --profile benchmark --run-name l4-charb-cosine-20260511-02 --epochs 60 --train-batch-size 8 --eval-batch-size 2 --num-workers 6 --base-channels 32
```

Resume from a saved training checkpoint by pointing at the prior `latest` checkpoint. When resuming, `--epochs` is the total target epoch count, not the number of extra epochs:

```bash
.venv/bin/python -m modal run modal_app.py --profile benchmark --run-name l4-baseline-continue-20260424-01 --epochs 60 --base-channels 32 --resume-checkpoint results/checkpoints/l4-baseline-20260421-01_benchmark_unet_deblurring_latest.pt
```

If the resume checkpoint comes from an older `L1 + StepLR` run, the benchmark will automatically inherit that legacy recipe so the checkpoint remains compatible.

Download results from the results volume:

```bash
.venv/bin/python -m modal volume get image-deblurring-results / results/modal-download/
```
