"""Dataset utilities aligned with the final HALO DNANet checkpoints."""

from pathlib import Path
import random

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
from torch.utils.data import Dataset


DATASET_META = {
    "NUAA-SIRST": {"image_folder": "images", "split_method": "50_50"},
    "NUDT-SIRST": {"image_folder": "images", "split_method": "50_50"},
    "IRSTD-1k": {"image_folder": "IRSTD1k_Img", "split_method": "50_50"},
}


def get_default_image_folder(dataset_name):
    return DATASET_META.get(dataset_name, {}).get("image_folder", "images")


def get_default_split_method(dataset_name):
    return DATASET_META.get(dataset_name, {}).get("split_method", "50_50")


def resolve_dataset_root(root, dataset_name):
    root = Path(root)
    if root.name == dataset_name:
        return root
    return root / dataset_name


def read_id_file(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def resolve_split_files(root, dataset_name, split_method="50_50", split_dir=None):
    dataset_root = resolve_dataset_root(root, dataset_name)
    candidates = []
    if split_dir:
        candidates.append(Path(split_dir))
    # Use the repository split that matches the reported checkpoint evaluation.
    # It takes priority over similarly named splits bundled with dataset downloads.
    candidates.append(Path(__file__).resolve().parent / "splits" / dataset_name)
    candidates.append(dataset_root / split_method)
    candidates.append(dataset_root)

    for candidate in candidates:
        train_txt = candidate / "train.txt"
        test_txt = candidate / "test.txt"
        if train_txt.exists() and test_txt.exists():
            return train_txt, test_txt
    raise FileNotFoundError(f"Could not find split files for {dataset_name}")


def load_split_ids(root, dataset_name, split_method="50_50", split_dir=None):
    train_txt, test_txt = resolve_split_files(root, dataset_name, split_method, split_dir)
    return read_id_file(train_txt), read_id_file(test_txt)


def load_param(channel_size="three", backbone="resnet_18"):
    nb_filter_map = {
        "one": [4, 8, 16, 32, 64],
        "two": [8, 16, 32, 64, 128],
        "three": [16, 32, 64, 128, 256],
        "four": [32, 64, 128, 256, 512],
    }
    num_blocks_map = {
        "resnet_10": [1, 1, 1, 1],
        "resnet_18": [2, 2, 2, 2],
        "resnet_34": [3, 4, 6, 3],
        "vgg_10": [1, 1, 1, 1],
    }
    return nb_filter_map[channel_size], num_blocks_map[backbone]


def _normalize_tensor(tensor, in_channels):
    if in_channels == 1:
        mean = torch.tensor([0.5], dtype=tensor.dtype).view(-1, 1, 1)
        std = torch.tensor([0.5], dtype=tensor.dtype).view(-1, 1, 1)
    else:
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(-1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(-1, 1, 1)
    return (tensor - mean) / std


def _pil_to_tensor(image, in_channels):
    array = np.array(image, dtype=np.float32)
    if in_channels == 1:
        tensor = torch.from_numpy(array / 255.0).unsqueeze(0)
    else:
        tensor = torch.from_numpy((array / 255.0).transpose(2, 0, 1))
    return _normalize_tensor(tensor, in_channels)


def _mask_to_tensor(mask):
    return torch.from_numpy(np.expand_dims(np.array(mask, dtype=np.float32) / 255.0, axis=0))


class TrainSetLoader(Dataset):
    def __init__(
        self,
        dataset_dir,
        img_id,
        base_size=256,
        crop_size=256,
        transform=None,
        suffix=".png",
        in_channels=1,
        image_folder="images",
        mask_folder="masks_sweep_sigma1.500_expand1.5",
        aug_rotate=False,
    ):
        self.transform = transform
        self._items = img_id
        self.dataset_dir = Path(dataset_dir)
        self.images = self.dataset_dir / image_folder
        self.masks = self.dataset_dir / mask_folder
        self.base_size = base_size
        self.crop_size = crop_size
        self.suffix = suffix
        self.in_channels = in_channels
        self.aug_rotate = aug_rotate

    def _load_image(self, path):
        if self.in_channels == 1:
            return Image.open(path).convert("L")
        return Image.open(path).convert("RGB")

    def _sync_transform(self, image, mask):
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        if self.aug_rotate:
            k = random.randint(0, 3)
            if k:
                rotation = {
                    1: Image.Transpose.ROTATE_90,
                    2: Image.Transpose.ROTATE_180,
                    3: Image.Transpose.ROTATE_270,
                }[k]
                image = image.transpose(rotation)
                mask = mask.transpose(rotation)

        long_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        width, height = image.size
        if height > width:
            new_height = long_size
            new_width = int(width * long_size / height + 0.5)
            short_size = new_width
        else:
            new_width = long_size
            new_height = int(height * long_size / width + 0.5)
            short_size = new_height

        image = image.resize((new_width, new_height), Image.BILINEAR)
        mask = mask.resize((new_width, new_height), Image.NEAREST)

        if short_size < self.crop_size:
            pad_h = self.crop_size - new_height if new_height < self.crop_size else 0
            pad_w = self.crop_size - new_width if new_width < self.crop_size else 0
            image = ImageOps.expand(image, border=(0, 0, pad_w, pad_h), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, pad_w, pad_h), fill=0)

        width, height = image.size
        x1 = random.randint(0, width - self.crop_size)
        y1 = random.randint(0, height - self.crop_size)
        image = image.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        if random.random() < 0.5:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.random()))

        return image, mask

    def __getitem__(self, idx):
        image_id = self._items[idx]
        image = self._load_image(self.images / f"{image_id}{self.suffix}")
        mask = Image.open(self.masks / f"{image_id}{self.suffix}").convert("L")
        image, mask = self._sync_transform(image, mask)
        image_tensor = _pil_to_tensor(image, self.in_channels)
        mask_tensor = _mask_to_tensor(mask)
        return image_tensor, mask_tensor

    def __len__(self):
        return len(self._items)


class TestSetLoader(Dataset):
    def __init__(
        self,
        dataset_dir,
        img_id,
        transform=None,
        base_size=256,
        crop_size=256,
        suffix=".png",
        in_channels=1,
        image_folder="images",
        mask_folder="masks",
    ):
        self.transform = transform
        self._items = img_id
        self.dataset_dir = Path(dataset_dir)
        self.images = self.dataset_dir / image_folder
        self.masks = self.dataset_dir / mask_folder
        self.base_size = base_size
        self.crop_size = crop_size
        self.suffix = suffix
        self.in_channels = in_channels

    def _load_image(self, path):
        if self.in_channels == 1:
            return Image.open(path).convert("L")
        return Image.open(path).convert("RGB")

    def __getitem__(self, idx):
        image_id = self._items[idx]
        image = self._load_image(self.images / f"{image_id}{self.suffix}")
        mask = Image.open(self.masks / f"{image_id}{self.suffix}").convert("L")
        image = image.resize((self.base_size, self.base_size), Image.BILINEAR)
        mask = mask.resize((self.base_size, self.base_size), Image.NEAREST)
        image_tensor = _pil_to_tensor(image, self.in_channels)
        mask_tensor = _mask_to_tensor(mask)
        return image_tensor, mask_tensor, image_id

    def __len__(self):
        return len(self._items)
