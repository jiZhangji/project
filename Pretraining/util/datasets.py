from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


class SARImageDataset(Dataset):
    """Recursively loads single-channel SAR images from a directory."""

    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {self.root}")
        self.images = sorted(
            path for path in self.root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.images:
            raise RuntimeError(f"No supported image files found under: {self.root}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        with Image.open(self.images[index]) as image:
            image = image.convert("L")
            if self.transform is not None:
                image = self.transform(image)
        return image, 0


class SyntheticSARDataset(Dataset):
    """Generated inputs used only for smoke tests and memory probing."""

    def __init__(self, length=65536, image_size=224):
        self.length = int(length)
        self.image_size = int(image_size)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(index)
        image = torch.rand((1, self.image_size, self.image_size), generator=generator)
        return image, 0


def load_data(file_dir, transform):
    return SARImageDataset(file_dir, transform=transform)
