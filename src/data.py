from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple, Union

from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .transforms import build_eval_transform, build_train_transform
from .utils import find_dataset_root, pair_image_files


@dataclass(frozen=True)
class PairRecord:
    split: str
    source_split: str
    sequence: str
    filename: str
    blur_path: Path
    sharp_path: Path


@dataclass(frozen=True)
class SequenceSplit:
    train_sequences: list[str]
    val_sequences: list[str]
    test_sequences: list[str]
    seed: int
    val_fraction: float


def _list_sequence_names(split_dir: Path) -> list[str]:
    if not split_dir.is_dir():
        return []
    return sorted(path.name for path in split_dir.iterdir() if path.is_dir())


def split_train_validation_sequences(
    sequence_names: Sequence[str],
    *,
    val_fraction: float = 0.10,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must lie in [0, 1)")

    unique_names = sorted(dict.fromkeys(sequence_names))
    if not unique_names:
        return [], []
    if val_fraction == 0.0:
        return unique_names, []

    shuffled = unique_names[:]
    random.Random(seed).shuffle(shuffled)
    num_val = max(1, math.ceil(len(unique_names) * val_fraction))
    val_names = set(shuffled[:num_val])
    train_names = [name for name in unique_names if name not in val_names]
    validation_names = [name for name in unique_names if name in val_names]
    return train_names, validation_names


def build_sequence_split(
    *,
    root: Optional[Union[str, Path]] = None,
    val_fraction: float = 0.10,
    seed: int = 42,
) -> SequenceSplit:
    dataset_root = find_dataset_root(root)
    train_names, val_names = split_train_validation_sequences(
        _list_sequence_names(dataset_root / "train"),
        val_fraction=val_fraction,
        seed=seed,
    )
    test_names = _list_sequence_names(dataset_root / "test")
    return SequenceSplit(
        train_sequences=train_names,
        val_sequences=val_names,
        test_sequences=test_names,
        seed=seed,
        val_fraction=val_fraction,
    )


class GoProPairDataset(Dataset):
    def __init__(
        self,
        *,
        root: Optional[Union[str, Path]] = None,
        split: str = "train",
        mode: Optional[str] = None,
        blur_dir_name: str = "blur",
        sharp_dir_name: str = "sharp",
        sequences: Optional[Sequence[str]] = None,
        transform=None,
        patch_size: Optional[Union[int, Tuple[int, int]]] = 256,
        eval_center_crop: Optional[Union[int, Tuple[int, int]]] = None,
        eval_resize_shorter_side: Optional[int] = None,
        eval_resize: Optional[Union[int, Tuple[int, int]]] = None,
        val_fraction: float = 0.10,
        seed: int = 42,
        include_metadata: bool = True,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of {'train', 'val', 'test'}")

        self.root = find_dataset_root(root)
        self.split = split
        self.mode = mode or ("train" if split == "train" else "eval")
        if self.mode not in {"train", "eval"}:
            raise ValueError("mode must be 'train' or 'eval'")

        self.blur_dir_name = blur_dir_name
        self.sharp_dir_name = sharp_dir_name
        self.include_metadata = include_metadata
        self.val_fraction = val_fraction
        self.seed = seed
        self.source_split, self.sequence_names = self._resolve_sequences(split, sequences)
        self.records = self._build_records()
        self.transform = transform or self._build_default_transform(
            patch_size=patch_size,
            eval_center_crop=eval_center_crop,
            eval_resize_shorter_side=eval_resize_shorter_side,
            eval_resize=eval_resize,
        )

    def _resolve_sequences(
        self,
        split: str,
        sequences: Optional[Sequence[str]],
    ) -> tuple[str, list[str]]:
        if split == "test":
            source_split = "test"
            available_sequences = _list_sequence_names(self.root / source_split)
        elif (self.root / "val").is_dir():
            source_split = split
            available_sequences = _list_sequence_names(self.root / source_split)
        else:
            source_split = "train"
            train_sequences, validation_sequences = split_train_validation_sequences(
                _list_sequence_names(self.root / "train"),
                val_fraction=self.val_fraction,
                seed=self.seed,
            )
            available_sequences = train_sequences if split == "train" else validation_sequences
            if split == "val" and not available_sequences:
                raise ValueError("Validation split is empty. Increase val_fraction or add a dedicated val folder.")

        if sequences is None:
            return source_split, available_sequences

        requested_sequences = list(dict.fromkeys(sequences))
        missing_sequences = sorted(set(requested_sequences) - set(available_sequences))
        if missing_sequences:
            raise ValueError(f"Requested sequences are not available for split '{split}': {missing_sequences}")

        return source_split, requested_sequences

    def _build_records(self) -> list[PairRecord]:
        split_dir = self.root / self.source_split
        records: list[PairRecord] = []

        for sequence_name in self.sequence_names:
            sequence_dir = split_dir / sequence_name
            paired_files = pair_image_files(sequence_dir / self.blur_dir_name, sequence_dir / self.sharp_dir_name)
            for blur_path, sharp_path in paired_files:
                records.append(
                    PairRecord(
                        split=self.split,
                        source_split=self.source_split,
                        sequence=sequence_name,
                        filename=blur_path.name,
                        blur_path=blur_path,
                        sharp_path=sharp_path,
                    )
                )

        return records

    def _build_default_transform(
        self,
        *,
        patch_size: Optional[Union[int, Tuple[int, int]]],
        eval_center_crop: Optional[Union[int, Tuple[int, int]]],
        eval_resize_shorter_side: Optional[int],
        eval_resize: Optional[Union[int, Tuple[int, int]]],
    ):
        if self.mode == "train":
            return build_train_transform(crop_size=patch_size)

        return build_eval_transform(
            resize_shorter_side=eval_resize_shorter_side,
            resize_size=eval_resize,
            center_crop_size=eval_center_crop,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        with Image.open(record.blur_path) as blur_image:
            blur = blur_image.convert("RGB")
        with Image.open(record.sharp_path) as sharp_image:
            sharp = sharp_image.convert("RGB")

        if blur.size != sharp.size:
            raise ValueError(f"Pair size mismatch for {record.sequence}/{record.filename}: {blur.size} vs {sharp.size}")

        if self.transform is not None:
            blur, sharp = self.transform(blur, sharp)

        sample: dict[str, Any] = {
            "blur": blur,
            "sharp": sharp,
        }
        if self.include_metadata:
            sample["metadata"] = {
                "split": record.split,
                "source_split": record.source_split,
                "sequence": record.sequence,
                "filename": record.filename,
                "blur_path": str(record.blur_path),
                "sharp_path": str(record.sharp_path),
            }
        return sample


def build_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: Optional[bool] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: Optional[bool] = None,
    persistent_workers: bool = False,
    prefetch_factor: Optional[int] = None,
) -> DataLoader:
    if shuffle is None:
        shuffle = getattr(dataset, "mode", "eval") == "train"
    if drop_last is None:
        drop_last = getattr(dataset, "mode", "eval") == "train"

    dataloader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
    }
    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            dataloader_kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(
        **dataloader_kwargs,
    )


def build_default_datasets(
    *,
    root: Optional[Union[str, Path]] = None,
    patch_size: Optional[Union[int, Tuple[int, int]]] = 256,
    val_fraction: float = 0.10,
    seed: int = 42,
    blur_dir_name: str = "blur",
    sharp_dir_name: str = "sharp",
    eval_center_crop: Optional[Union[int, Tuple[int, int]]] = None,
    eval_resize_shorter_side: Optional[int] = None,
    eval_resize: Optional[Union[int, Tuple[int, int]]] = None,
) -> dict[str, GoProPairDataset]:
    train_dataset = GoProPairDataset(
        root=root,
        split="train",
        mode="train",
        patch_size=patch_size,
        blur_dir_name=blur_dir_name,
        sharp_dir_name=sharp_dir_name,
        val_fraction=val_fraction,
        seed=seed,
    )

    validation_dataset = GoProPairDataset(
        root=root,
        split="val",
        mode="eval",
        patch_size=None,
        eval_center_crop=eval_center_crop,
        eval_resize_shorter_side=eval_resize_shorter_side,
        eval_resize=eval_resize,
        blur_dir_name=blur_dir_name,
        sharp_dir_name=sharp_dir_name,
        val_fraction=val_fraction,
        seed=seed,
    )

    test_dataset = GoProPairDataset(
        root=root,
        split="test",
        mode="eval",
        patch_size=None,
        eval_center_crop=eval_center_crop,
        eval_resize_shorter_side=eval_resize_shorter_side,
        eval_resize=eval_resize,
        blur_dir_name=blur_dir_name,
        sharp_dir_name=sharp_dir_name,
        val_fraction=val_fraction,
        seed=seed,
    )

    return {
        "train": train_dataset,
        "val": validation_dataset,
        "test": test_dataset,
    }


__all__ = [
    "GoProPairDataset",
    "PairRecord",
    "SequenceSplit",
    "build_sequence_split",
    "build_dataloader",
    "build_default_datasets",
    "split_train_validation_sequences",
]
