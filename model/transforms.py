"""Image transforms for training and evaluation.

Input images are 256x256 face crops from model/faces.py.
"""

from __future__ import annotations

import io
import random

from PIL import Image, ImageFilter
from torchvision import transforms

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class RandomJPEG:
    """Re-encode as JPEG at a random quality.

    Images on the internet are compressed over and over, and compression wipes
    out exactly the fine-grained artifacts a detector tends to rely on.
    Training with it makes the model less brittle on real uploads.
    """

    def __init__(self, quality_range: tuple[int, int] = (40, 95), p: float = 0.5) -> None:
        self.quality_range = quality_range
        self.p = p

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return image
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=random.randint(*self.quality_range))
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


class RandomDownscale:
    """Shrink then re-enlarge, simulating low-resolution uploads."""

    def __init__(self, scale_range: tuple[float, float] = (0.4, 0.9), p: float = 0.3) -> None:
        self.scale_range = scale_range
        self.p = p

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return image
        w, h = image.size
        s = random.uniform(*self.scale_range)
        small = image.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)


class RandomBlur:
    def __init__(self, radius_range: tuple[float, float] = (0.3, 1.5), p: float = 0.2) -> None:
        self.radius_range = radius_range
        self.p = p

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return image
        return image.filter(ImageFilter.GaussianBlur(random.uniform(*self.radius_range)))


def get_train_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            RandomJPEG(),
            RandomDownscale(),
            RandomBlur(),
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def get_eval_transforms() -> transforms.Compose:
    # Resize the whole crop instead of center-cropping, so the face edges
    # (where swap artifacts often live) are never cut off.
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
