import os
import random
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLASS_NAMES = ("dog",)

class Part2Dataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str,
        transform=None,
        augment: bool = False,
        max_samples: Optional[int] = None,
    ) -> None:
        self.root = os.path.join(root_dir, split)
        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"Directory not found: {self.root}")
        self.transform = transform
        self.augment = augment
        self.samples = self._load_samples(max_samples=max_samples)

    def _load_samples(self, max_samples: Optional[int] = None) -> List[Dict[str, str]]:
        samples: List[Dict[str, str]] = []
        for file_name in sorted(os.listdir(self.root)):
            if not file_name.lower().endswith(".jpg"):
                continue
            image_path = os.path.join(self.root, file_name)
            annotation_path = os.path.join(self.root, file_name.rsplit(".", 1)[0] + ".xml")
            if os.path.exists(annotation_path):
                samples.append({"image_path": image_path, "annotation_path": annotation_path})
        if max_samples is not None:
            return samples[:max_samples]
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        image = Image.open(sample["image_path"]).convert("RGB")
        box = self._load_first_box(sample["annotation_path"])
        width, height = image.size

        target = torch.tensor(
            [
                box[0] / width,
                box[1] / height,
                box[2] / width,
                box[3] / height,
            ],
            dtype=torch.float32,
        )

        if self.augment:
            if random.random() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                target = torch.tensor(
                    [1.0 - target[2], target[1], 1.0 - target[0], target[3]],
                    dtype=torch.float32,
                )
            if random.random() < 0.5:
                image = transforms.functional.adjust_brightness(image, random.uniform(0.8, 1.2))
            if random.random() < 0.5:
                image = transforms.functional.adjust_contrast(image, random.uniform(0.8, 1.2))
            if random.random() < 0.5:
                image = transforms.functional.adjust_saturation(image, random.uniform(0.8, 1.2))

        if self.transform:
            image = self.transform(image)
        return image, target

    @staticmethod
    def _load_first_box(annotation_path: str) -> torch.Tensor:
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        object_node = root.find("object")
        if object_node is None:
            raise ValueError(f"No object annotation found in {annotation_path}")
        bbox = object_node.find("bndbox")
        if bbox is None:
            raise ValueError(f"No bounding box found in {annotation_path}")
        xmin = float(bbox.findtext("xmin", default="0"))
        ymin = float(bbox.findtext("ymin", default="0"))
        xmax = float(bbox.findtext("xmax", default="0"))
        ymax = float(bbox.findtext("ymax", default="0"))
        return torch.tensor([xmin, ymin, xmax, ymax], dtype=torch.float32)


def build_train_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_eval_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def create_dataloader(
    dataset_root: str,
    split: str,
    batch_size: int,
    image_size: int = 224,
    train: bool = True,
    shuffle: Optional[bool] = None,
    num_workers: int = 0,
    max_samples: Optional[int] = None,
) -> DataLoader:
    transform = build_train_transform(image_size) if train else build_eval_transform(image_size)
    dataset = Part2Dataset(
        root_dir=dataset_root,
        split=split,
        transform=transform,
        augment=train,
        max_samples=max_samples,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train if shuffle is None else shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
