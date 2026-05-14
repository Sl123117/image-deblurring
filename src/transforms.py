from __future__ import annotations

import math
import random
from typing import Iterable, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

SizeLike = Union[int, Tuple[int, int]]


def _normalize_size(size: SizeLike) -> tuple[int, int]:
    if isinstance(size, int):
        if size <= 0:
            raise ValueError("size must be positive")
        return size, size

    if len(size) != 2:
        raise ValueError("size must be an int or a (height, width) tuple")

    height, width = int(size[0]), int(size[1])
    if height <= 0 or width <= 0:
        raise ValueError("size dimensions must be positive")
    return height, width


def _validate_same_size(blur: Image.Image, sharp: Image.Image) -> tuple[int, int]:
    if blur.size != sharp.size:
        raise ValueError(f"blur and sharp images must share the same size, got {blur.size} and {sharp.size}")
    return blur.size


def _pil_to_float_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 2:
        array = array[..., None]
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor / 255.0


class ComposePair:
    def __init__(self, transforms: Iterable):
        self.transforms = [transform for transform in transforms if transform is not None]

    def __call__(self, blur: Image.Image, sharp: Image.Image):
        for transform in self.transforms:
            blur, sharp = transform(blur, sharp)
        return blur, sharp


class RandomCropPair:
    def __init__(self, size: SizeLike):
        self.height, self.width = _normalize_size(size)

    def __call__(self, blur: Image.Image, sharp: Image.Image):
        image_width, image_height = _validate_same_size(blur, sharp)
        if image_height < self.height or image_width < self.width:
            raise ValueError(
                f"crop size {(self.height, self.width)} is larger than image size {(image_height, image_width)}"
            )

        top = random.randint(0, image_height - self.height)
        left = random.randint(0, image_width - self.width)
        box = (left, top, left + self.width, top + self.height)
        return blur.crop(box), sharp.crop(box)


class CenterCropPair:
    def __init__(self, size: SizeLike):
        self.height, self.width = _normalize_size(size)

    def __call__(self, blur: Image.Image, sharp: Image.Image):
        image_width, image_height = _validate_same_size(blur, sharp)
        if image_height < self.height or image_width < self.width:
            raise ValueError(
                f"center crop size {(self.height, self.width)} is larger than image size {(image_height, image_width)}"
            )

        top = (image_height - self.height) // 2
        left = (image_width - self.width) // 2
        box = (left, top, left + self.width, top + self.height)
        return blur.crop(box), sharp.crop(box)


class RandomHorizontalFlipPair:
    def __init__(self, probability: float = 0.5):
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must lie in [0, 1]")
        self.probability = probability

    def __call__(self, blur: Image.Image, sharp: Image.Image):
        _validate_same_size(blur, sharp)
        if random.random() < self.probability:
            return blur.transpose(Image.FLIP_LEFT_RIGHT), sharp.transpose(Image.FLIP_LEFT_RIGHT)
        return blur, sharp


class ResizeShortestSidePair:
    def __init__(self, shorter_side: int, resample: int = Image.BICUBIC):
        if shorter_side <= 0:
            raise ValueError("shorter_side must be positive")
        self.shorter_side = shorter_side
        self.resample = resample

    def __call__(self, blur: Image.Image, sharp: Image.Image):
        image_width, image_height = _validate_same_size(blur, sharp)
        scale = self.shorter_side / min(image_width, image_height)
        new_width = max(1, int(round(image_width * scale)))
        new_height = max(1, int(round(image_height * scale)))
        size = (new_width, new_height)
        return blur.resize(size, self.resample), sharp.resize(size, self.resample)


class ResizePair:
    def __init__(
        self,
        size: SizeLike,
        *,
        allow_aspect_distortion: bool = False,
        resample: int = Image.BICUBIC,
    ):
        self.height, self.width = _normalize_size(size)
        self.allow_aspect_distortion = allow_aspect_distortion
        self.resample = resample

    def __call__(self, blur: Image.Image, sharp: Image.Image):
        image_width, image_height = _validate_same_size(blur, sharp)
        if not self.allow_aspect_distortion:
            source_aspect = image_width / image_height
            target_aspect = self.width / self.height
            if not math.isclose(source_aspect, target_aspect, rel_tol=1e-3, abs_tol=1e-3):
                raise ValueError(
                    "resize would distort aspect ratio; use ResizeShortestSidePair or set "
                    "allow_aspect_distortion=True explicitly"
                )

        size = (self.width, self.height)
        return blur.resize(size, self.resample), sharp.resize(size, self.resample)


class ToTensorPair:
    def __call__(self, blur: Image.Image, sharp: Image.Image):
        _validate_same_size(blur, sharp)
        return _pil_to_float_tensor(blur), _pil_to_float_tensor(sharp)


def build_train_transform(
    *,
    crop_size: Optional[SizeLike] = 256,
    horizontal_flip_probability: float = 0.5,
    to_tensor: bool = True,
):
    transforms = []
    if crop_size is not None:
        transforms.append(RandomCropPair(crop_size))
    if horizontal_flip_probability > 0.0:
        transforms.append(RandomHorizontalFlipPair(horizontal_flip_probability))
    if to_tensor:
        transforms.append(ToTensorPair())
    return ComposePair(transforms)


def build_eval_transform(
    *,
    resize_shorter_side: Optional[int] = None,
    resize_size: Optional[SizeLike] = None,
    center_crop_size: Optional[SizeLike] = None,
    to_tensor: bool = True,
):
    transforms = []
    if resize_shorter_side is not None:
        transforms.append(ResizeShortestSidePair(resize_shorter_side))
    if resize_size is not None:
        transforms.append(ResizePair(resize_size))
    if center_crop_size is not None:
        transforms.append(CenterCropPair(center_crop_size))
    if to_tensor:
        transforms.append(ToTensorPair())
    return ComposePair(transforms)


__all__ = [
    "CenterCropPair",
    "ComposePair",
    "RandomCropPair",
    "RandomHorizontalFlipPair",
    "ResizePair",
    "ResizeShortestSidePair",
    "ToTensorPair",
    "build_eval_transform",
    "build_train_transform",
]
