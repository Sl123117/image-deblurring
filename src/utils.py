from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Optional, Union

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


class PairConsistencyError(ValueError):
    pass


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def find_dataset_root(root: Optional[Union[str, Path]] = None) -> Path:
    project_root = get_project_root()
    candidates: list[Path] = []

    if root is not None:
        candidates.append(Path(root))

    candidates.extend(
        [
            project_root / "data" / "raw" / "GOPRO_Large",
            project_root / "GOPRO_Large",
            project_root / "data" / "raw",
        ]
    )

    for candidate in candidates:
        if (candidate / "train").is_dir() and (candidate / "test").is_dir():
            return candidate.resolve()

    expected_roots = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find the GoPro dataset. Checked: {expected_roots}")


def list_image_files(folder: Union[str, Path]) -> list[Path]:
    folder_path = Path(folder)
    return sorted(path for path in folder_path.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def pair_image_files(
    blur_dir: Union[str, Path],
    sharp_dir: Union[str, Path],
) -> list[tuple[Path, Path]]:
    blur_path = Path(blur_dir)
    sharp_path = Path(sharp_dir)

    if not blur_path.is_dir():
        raise PairConsistencyError(f"Missing blur directory: {blur_path}")
    if not sharp_path.is_dir():
        raise PairConsistencyError(f"Missing sharp directory: {sharp_path}")

    blur_files = {path.name: path for path in list_image_files(blur_path)}
    sharp_files = {path.name: path for path in list_image_files(sharp_path)}
    missing_in_sharp = sorted(set(blur_files) - set(sharp_files))
    missing_in_blur = sorted(set(sharp_files) - set(blur_files))

    if missing_in_sharp or missing_in_blur:
        snippets = []
        if missing_in_sharp:
            snippets.append(f"missing in sharp: {missing_in_sharp[:5]}")
        if missing_in_blur:
            snippets.append(f"missing in blur: {missing_in_blur[:5]}")
        raise PairConsistencyError(f"Pair mismatch between {blur_path} and {sharp_path} ({'; '.join(snippets)})")

    return [(blur_files[name], sharp_files[name]) for name in sorted(blur_files)]


def audit_gopro_dataset(
    root: Optional[Union[str, Path]] = None,
    *,
    blur_dir_name: str = "blur",
    sharp_dir_name: str = "sharp",
) -> dict[str, Any]:
    dataset_root = find_dataset_root(root)
    split_summaries: dict[str, Any] = {}
    global_size_counter: Counter[tuple[int, int]] = Counter()
    issues: list[str] = []
    auxiliary_input_dirs: Counter[str] = Counter()
    total_pairs = 0
    total_sequences = 0

    for split_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        split_pair_count = 0
        split_size_counter: Counter[tuple[int, int]] = Counter()
        sequence_summaries = []

        for sequence_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            available_dirs = sorted(path.name for path in sequence_dir.iterdir() if path.is_dir())
            for directory_name in available_dirs:
                if directory_name not in {blur_dir_name, sharp_dir_name}:
                    auxiliary_input_dirs[directory_name] += 1

            try:
                pairs = pair_image_files(sequence_dir / blur_dir_name, sequence_dir / sharp_dir_name)
            except PairConsistencyError as error:
                issues.append(f"{split_dir.name}/{sequence_dir.name}: {error}")
                continue

            sequence_size_counter: Counter[tuple[int, int]] = Counter()
            sequence_issues: list[str] = []

            for blur_path, sharp_path in pairs:
                with Image.open(blur_path) as blur_image, Image.open(sharp_path) as sharp_image:
                    blur_size = blur_image.size
                    sharp_size = sharp_image.size

                if blur_size != sharp_size:
                    message = (
                        f"{split_dir.name}/{sequence_dir.name}/{blur_path.name}: "
                        f"blur size {blur_size} does not match sharp size {sharp_size}"
                    )
                    sequence_issues.append(message)
                    issues.append(message)
                    continue

                sequence_size_counter[blur_size] += 1
                split_size_counter[blur_size] += 1
                global_size_counter[blur_size] += 1
                split_pair_count += 1
                total_pairs += 1

            total_sequences += 1
            sequence_summaries.append(
                {
                    "sequence": sequence_dir.name,
                    "pair_count": sum(sequence_size_counter.values()),
                    "image_sizes": {f"{width}x{height}": count for (width, height), count in sorted(sequence_size_counter.items())},
                    "available_dirs": available_dirs,
                    "issues": sequence_issues,
                }
            )

        split_summaries[split_dir.name] = {
            "sequence_count": len(sequence_summaries),
            "pair_count": split_pair_count,
            "image_sizes": {f"{width}x{height}": count for (width, height), count in sorted(split_size_counter.items())},
            "sequences": sequence_summaries,
        }

    return {
        "dataset_name": dataset_root.name,
        "dataset_root": str(dataset_root),
        "split_names": sorted(split_summaries),
        "total_sequences": total_sequences,
        "total_pairs": total_pairs,
        "image_sizes": {f"{width}x{height}": count for (width, height), count in sorted(global_size_counter.items())},
        "auxiliary_input_dirs": dict(sorted(auxiliary_input_dirs.items())),
        "splits": split_summaries,
        "issues": issues,
    }


def format_dataset_report(
    audit: dict[str, Any],
    *,
    validation_strategy: str = "Create validation data by holding out whole training sequences with a fixed random seed.",
    validation_fraction: float = 0.10,
    validation_seed: int = 42,
) -> str:
    lines = [
        "# Dataset Inspection Report",
        "",
        f"- Dataset found: {audit['dataset_name']}",
        f"- Dataset root: `{audit['dataset_root']}`",
        f"- Total verified pairs: {audit['total_pairs']}",
        f"- Total sequences: {audit['total_sequences']}",
        f"- Image sizes: {audit['image_sizes']}",
        f"- Auxiliary per-sequence directories: {audit['auxiliary_input_dirs'] or 'None'}",
        "",
        "## Split Summary",
        "",
    ]

    for split_name in audit["split_names"]:
        split_summary = audit["splits"][split_name]
        lines.append(
            f"- `{split_name}`: {split_summary['sequence_count']} sequences, {split_summary['pair_count']} verified blur/sharp pairs, sizes {split_summary['image_sizes']}"
        )

    lines.extend(
        [
            "",
            "## Pairing Logic",
            "",
            "- Each sequence uses `blur/<filename>` paired with `sharp/<filename>`.",
            "- Pair matching is one-to-one by identical filename within the same sequence directory.",
            "",
            "## Validation Recommendation",
            "",
            f"- {validation_strategy}",
            f"- Suggested holdout fraction: {validation_fraction:.0%} of training sequences, with seed `{validation_seed}`.",
        ]
    )

    if audit["issues"]:
        lines.extend(["", "## Issues", ""])
        lines.extend(f"- {issue}" for issue in audit["issues"])
    else:
        lines.extend(["", "## Issues", "", "- No missing pairs or resolution mismatches were found in the verified data."])

    return "\n".join(lines) + "\n"


def write_dataset_report(
    report_path: Union[str, Path],
    audit: dict[str, Any],
    *,
    validation_strategy: str = "Create validation data by holding out whole training sequences with a fixed random seed.",
    validation_fraction: float = 0.10,
    validation_seed: int = 42,
) -> Path:
    output_path = Path(report_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        format_dataset_report(
            audit,
            validation_strategy=validation_strategy,
            validation_fraction=validation_fraction,
            validation_seed=validation_seed,
        ),
        encoding="utf-8",
    )
    return output_path


def save_sample_visualizations(
    output_dir: Union[str, Path],
    *,
    root: Optional[Union[str, Path]] = None,
    blur_dir_name: str = "blur",
    sharp_dir_name: str = "sharp",
    num_samples: int = 4,
) -> list[Path]:
    dataset_root = find_dataset_root(root)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    split_sequence_dirs = []
    for split_name in ("train", "test"):
        split_dir = dataset_root / split_name
        if split_dir.is_dir():
            split_sequence_dirs.extend((split_name, sequence_dir) for sequence_dir in sorted(split_dir.iterdir()) if sequence_dir.is_dir())

    if not split_sequence_dirs:
        raise FileNotFoundError(f"No sequence directories found under {dataset_root}")

    sample_count = min(num_samples, len(split_sequence_dirs))
    selected_indices = np.linspace(0, len(split_sequence_dirs) - 1, num=sample_count, dtype=int)
    saved_paths: list[Path] = []

    for index in selected_indices:
        split_name, sequence_dir = split_sequence_dirs[index]
        pairs = pair_image_files(sequence_dir / blur_dir_name, sequence_dir / sharp_dir_name)
        blur_path, sharp_path = pairs[len(pairs) // 2]

        with Image.open(blur_path) as blur_image:
            blur_array = np.asarray(blur_image.convert("RGB"))
        with Image.open(sharp_path) as sharp_image:
            sharp_array = np.asarray(sharp_image.convert("RGB"))

        figure, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        axes[0].imshow(blur_array)
        axes[0].set_title(f"{split_name}/{sequence_dir.name} blur")
        axes[1].imshow(sharp_array)
        axes[1].set_title(f"{split_name}/{sequence_dir.name} sharp")

        for axis in axes:
            axis.axis("off")

        output_path = destination / f"{split_name}_{sequence_dir.name}_{blur_path.stem}.png"
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(figure)
        saved_paths.append(output_path)

    return saved_paths


__all__ = [
    "PairConsistencyError",
    "audit_gopro_dataset",
    "find_dataset_root",
    "format_dataset_report",
    "pair_image_files",
    "save_sample_visualizations",
    "write_dataset_report",
]
