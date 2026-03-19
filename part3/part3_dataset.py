import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision.transforms import functional as TF


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLASS_NAMES = ("dog", "person")
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.reshape(0, 4)
    x1y1 = boxes[:, :2]
    x2y2 = boxes[:, :2] + boxes[:, 2:]
    return torch.cat((x1y1, x2y2), dim=1)


def xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.reshape(0, 4)
    centers = (boxes[:, :2] + boxes[:, 2:]) * 0.5
    sizes = boxes[:, 2:] - boxes[:, :2]
    return torch.cat((centers, sizes), dim=1)


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.reshape(0, 4)
    half_sizes = boxes[:, 2:] * 0.5
    top_left = boxes[:, :2] - half_sizes
    bottom_right = boxes[:, :2] + half_sizes
    return torch.cat((top_left, bottom_right), dim=1)


def clamp_boxes_xyxy(boxes: torch.Tensor, width: int, height: int) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.reshape(0, 4)
    clipped = boxes.clone()
    clipped[:, 0::2] = clipped[:, 0::2].clamp(0.0, float(width))
    clipped[:, 1::2] = clipped[:, 1::2].clamp(0.0, float(height))
    return clipped


def normalize_cxcywh(boxes: torch.Tensor, image_size: int) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.reshape(0, 4)
    scale = torch.tensor(
        [image_size, image_size, image_size, image_size],
        dtype=boxes.dtype,
        device=boxes.device,
    )
    return boxes / scale


def denormalize_xyxy(boxes: torch.Tensor, image_size: int) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.reshape(0, 4)
    scale = torch.tensor(
        [image_size, image_size, image_size, image_size],
        dtype=boxes.dtype,
        device=boxes.device,
    )
    return boxes * scale


@dataclass
class LetterboxMetadata:
    original_width: int
    original_height: int
    scaled_width: int
    scaled_height: int
    offset_x: int
    offset_y: int
    scale: float
    canvas_size: int


def _paste_on_canvas(
    resized: Image.Image,
    canvas_size: int,
    offset_x: int,
    offset_y: int,
    fill: Tuple[int, int, int] = (114, 114, 114),
) -> Image.Image:
    canvas = Image.new("RGB", (canvas_size, canvas_size), fill)
    src_x1 = max(0, -offset_x)
    src_y1 = max(0, -offset_y)
    dst_x1 = max(0, offset_x)
    dst_y1 = max(0, offset_y)
    copy_width = min(resized.width - src_x1, canvas_size - dst_x1)
    copy_height = min(resized.height - src_y1, canvas_size - dst_y1)
    if copy_width > 0 and copy_height > 0:
        crop = resized.crop((src_x1, src_y1, src_x1 + copy_width, src_y1 + copy_height))
        canvas.paste(crop, (dst_x1, dst_y1))
    return canvas


def letterbox_image_and_boxes(
    image: Image.Image,
    boxes: torch.Tensor,
    image_size: int,
    scale_multiplier: float = 1.0,
    random_offset: bool = False,
    fill: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[Image.Image, torch.Tensor, LetterboxMetadata]:
    original_width, original_height = image.size
    base_scale = min(image_size / original_width, image_size / original_height)
    scale = base_scale * scale_multiplier
    scaled_width = max(1, int(round(original_width * scale)))
    scaled_height = max(1, int(round(original_height * scale)))
    resized = image.resize((scaled_width, scaled_height), Image.BILINEAR)

    if random_offset:
        min_x = min(0, image_size - scaled_width)
        max_x = max(0, image_size - scaled_width)
        min_y = min(0, image_size - scaled_height)
        max_y = max(0, image_size - scaled_height)
        offset_x = random.randint(min_x, max_x)
        offset_y = random.randint(min_y, max_y)
    else:
        offset_x = (image_size - scaled_width) // 2
        offset_y = (image_size - scaled_height) // 2

    canvas = _paste_on_canvas(resized, image_size, offset_x, offset_y, fill=fill)
    transformed_boxes = boxes.clone()
    if transformed_boxes.numel() > 0:
        transformed_boxes = transformed_boxes * scale
        transformed_boxes[:, 0::2] += float(offset_x)
        transformed_boxes[:, 1::2] += float(offset_y)
        transformed_boxes = clamp_boxes_xyxy(transformed_boxes, image_size, image_size)

    meta = LetterboxMetadata(
        original_width=original_width,
        original_height=original_height,
        scaled_width=scaled_width,
        scaled_height=scaled_height,
        offset_x=offset_x,
        offset_y=offset_y,
        scale=scale,
        canvas_size=image_size,
    )
    return canvas, transformed_boxes, meta


def project_boxes_to_original_image(
    boxes_xyxy: torch.Tensor,
    meta: LetterboxMetadata,
) -> torch.Tensor:
    if boxes_xyxy.numel() == 0:
        return boxes_xyxy.reshape(0, 4)
    boxes = boxes_xyxy.clone()
    boxes[:, 0::2] -= float(meta.offset_x)
    boxes[:, 1::2] -= float(meta.offset_y)
    boxes = boxes / max(meta.scale, 1e-6)
    return clamp_boxes_xyxy(boxes, meta.original_width, meta.original_height)


class DetectionAugmentation:
    def __init__(self, image_size: int = 512, train: bool = True) -> None:
        self.image_size = image_size
        self.train = train

    def __call__(
        self,
        image: Image.Image,
        boxes: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, LetterboxMetadata]:
        original_image = image.copy()
        original_boxes = boxes.clone()
        original_labels = labels.clone()

        if self.train:
            image, boxes = self._maybe_horizontal_flip(image, boxes)
            image = self._photometric_distort(image)
            image = self._maybe_blur(image)
            image, boxes, meta = self._random_rescale_and_place(image, boxes)
        else:
            image, boxes, meta = letterbox_image_and_boxes(
                image=image,
                boxes=boxes,
                image_size=self.image_size,
                scale_multiplier=1.0,
                random_offset=False,
            )

        boxes, labels = self._filter_tiny_boxes(boxes, labels)
        if labels.numel() == 0:
            image, boxes, meta = letterbox_image_and_boxes(
                image=original_image,
                boxes=original_boxes,
                image_size=self.image_size,
                scale_multiplier=1.0,
                random_offset=False,
            )
            labels = original_labels

        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(image_tensor, IMAGENET_MEAN, IMAGENET_STD)
        abs_boxes = boxes.clone()
        norm_boxes = normalize_cxcywh(xyxy_to_cxcywh(boxes), self.image_size)
        return image_tensor, norm_boxes, labels, abs_boxes, meta

    def _maybe_horizontal_flip(
        self,
        image: Image.Image,
        boxes: torch.Tensor,
    ) -> Tuple[Image.Image, torch.Tensor]:
        if random.random() >= 0.5:
            return image, boxes
        width, _ = image.size
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        if boxes.numel() == 0:
            return image, boxes
        flipped = boxes.clone()
        flipped[:, 0] = width - boxes[:, 2]
        flipped[:, 2] = width - boxes[:, 0]
        return image, flipped

    def _photometric_distort(self, image: Image.Image) -> Image.Image:
        brightness = random.uniform(0.85, 1.15)
        contrast = random.uniform(0.85, 1.15)
        saturation = random.uniform(0.85, 1.15)
        image = ImageEnhance.Brightness(image).enhance(brightness)
        image = ImageEnhance.Contrast(image).enhance(contrast)
        image = ImageEnhance.Color(image).enhance(saturation)
        return image

    def _maybe_blur(self, image: Image.Image) -> Image.Image:
        if random.random() < 0.1:
            return image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 1.1)))
        return image

    def _random_rescale_and_place(
        self,
        image: Image.Image,
        boxes: torch.Tensor,
    ) -> Tuple[Image.Image, torch.Tensor, LetterboxMetadata]:
        for _ in range(5):
            scale_multiplier = random.uniform(0.9, 1.15)
            augmented_image, augmented_boxes, meta = letterbox_image_and_boxes(
                image=image,
                boxes=boxes,
                image_size=self.image_size,
                scale_multiplier=scale_multiplier,
                random_offset=True,
            )
            filtered_boxes, _ = self._filter_tiny_boxes(augmented_boxes, torch.zeros(len(augmented_boxes)))
            if filtered_boxes.numel() > 0:
                return augmented_image, augmented_boxes, meta
        return letterbox_image_and_boxes(
            image=image,
            boxes=boxes,
            image_size=self.image_size,
            scale_multiplier=1.0,
            random_offset=False,
        )

    @staticmethod
    def _filter_tiny_boxes(
        boxes: torch.Tensor,
        labels: torch.Tensor,
        min_size: float = 2.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if boxes.numel() == 0:
            return boxes.reshape(0, 4), labels[:0]
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        keep = (widths >= min_size) & (heights >= min_size)
        return boxes[keep], labels[keep]


class FixedSlotCocoDetectionDataset(Dataset):
    def __init__(
        self,
        dataset_root: str,
        split: str,
        image_size: int = 512,
        max_objects: int = 3,
        train: bool = True,
        max_samples: Optional[int] = None,
    ) -> None:
        self.dataset_root = dataset_root
        self.split = split
        self.image_dir = os.path.join(dataset_root, split)
        self.annotation_path = os.path.join(self.image_dir, "_annotations.coco.json")
        self.image_size = image_size
        self.max_objects = max_objects
        self.train = train
        self.transform = DetectionAugmentation(image_size=image_size, train=train)
        self.samples = self._load_samples_from_coco(
            annotation_path=self.annotation_path,
            image_dir=self.image_dir,
            max_samples=max_samples,
        )

    @staticmethod
    def _load_samples_from_coco(
        annotation_path: str,
        image_dir: str,
        max_samples: Optional[int] = None,
    ) -> List[Dict[str, object]]:
        with open(annotation_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        categories = {
            category["id"]: CLASS_TO_IDX[category["name"]]
            for category in data["categories"]
            if category["name"] in CLASS_TO_IDX
        }

        image_lookup = {image["id"]: image for image in data["images"]}
        annotations_by_image: Dict[int, List[Dict[str, object]]] = {
            image_id: [] for image_id in image_lookup
        }
        for annotation in data["annotations"]:
            category_id = annotation["category_id"]
            if category_id in categories:
                annotations_by_image[annotation["image_id"]].append(annotation)

        samples: List[Dict[str, object]] = []
        for image_id in sorted(image_lookup):
            image_record = image_lookup[image_id]
            image_annotations = annotations_by_image[image_id]
            if not image_annotations:
                continue
            samples.append(
                {
                    "image_id": image_id,
                    "path": os.path.join(image_dir, image_record["file_name"]),
                    "width": image_record["width"],
                    "height": image_record["height"],
                    "annotations": image_annotations,
                    "categories": categories,
                }
            )

        if max_samples is not None:
            return samples[:max_samples]
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        sample = self.samples[index]
        image = Image.open(sample["path"]).convert("RGB")

        categories: Dict[int, int] = sample["categories"]  # type: ignore[assignment]
        annotations: Sequence[Dict[str, object]] = sample["annotations"]  # type: ignore[assignment]
        boxes = torch.tensor(
            [annotation["bbox"] for annotation in annotations],
            dtype=torch.float32,
        )
        boxes = xywh_to_xyxy(boxes)
        labels = torch.tensor(
            [categories[annotation["category_id"]] for annotation in annotations],
            dtype=torch.long,
        )

        image_tensor, norm_boxes, labels, abs_boxes, meta = self.transform(image, boxes, labels)
        norm_boxes, labels, abs_boxes = self._sort_by_center(norm_boxes, labels, abs_boxes)

        num_objects = min(labels.shape[0], self.max_objects)
        target_boxes = torch.zeros((self.max_objects, 4), dtype=torch.float32)
        target_labels = torch.full((self.max_objects,), -1, dtype=torch.long)
        object_mask = torch.zeros((self.max_objects,), dtype=torch.float32)
        target_abs_boxes = torch.zeros((self.max_objects, 4), dtype=torch.float32)

        if num_objects > 0:
            target_boxes[:num_objects] = norm_boxes[:num_objects]
            target_labels[:num_objects] = labels[:num_objects]
            object_mask[:num_objects] = 1.0
            target_abs_boxes[:num_objects] = abs_boxes[:num_objects]

        target = {
            "boxes": target_boxes,
            "labels": target_labels,
            "object_mask": object_mask,
            "abs_boxes": target_abs_boxes,
            "image_id": torch.tensor(sample["image_id"], dtype=torch.long),
            "orig_size": torch.tensor(
                [meta.original_height, meta.original_width],
                dtype=torch.float32,
            ),
        }
        return image_tensor, target

    @staticmethod
    def _sort_by_center(
        norm_boxes: torch.Tensor,
        labels: torch.Tensor,
        abs_boxes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if norm_boxes.numel() == 0:
            return norm_boxes.reshape(0, 4), labels[:0], abs_boxes.reshape(0, 4)
        order = torch.argsort(norm_boxes[:, 0])
        return norm_boxes[order], labels[order], abs_boxes[order]


class GenericCocoDetectionDataset(FixedSlotCocoDetectionDataset):
    def __init__(
        self,
        annotation_path: str,
        image_dir: str,
        image_size: int = 512,
        max_objects: int = 3,
        train: bool = True,
        max_samples: Optional[int] = None,
    ) -> None:
        self.dataset_root = image_dir
        self.split = "generic"
        self.image_dir = image_dir
        self.annotation_path = annotation_path
        self.image_size = image_size
        self.max_objects = max_objects
        self.train = train
        self.transform = DetectionAugmentation(image_size=image_size, train=train)
        self.samples = self._load_samples_from_coco(
            annotation_path=self.annotation_path,
            image_dir=self.image_dir,
            max_samples=max_samples,
        )


def collate_detection_batch(
    batch: Sequence[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    images = torch.stack([item[0] for item in batch], dim=0)
    keys = batch[0][1].keys()
    targets = {key: torch.stack([item[1][key] for item in batch], dim=0) for key in keys}
    return images, targets


def create_dataloader(
    dataset_root: str,
    split: str,
    batch_size: int,
    image_size: int = 512,
    max_objects: int = 3,
    train: bool = True,
    shuffle: Optional[bool] = None,
    num_workers: int = 4,
    max_samples: Optional[int] = None,
) -> DataLoader:
    dataset = FixedSlotCocoDetectionDataset(
        dataset_root=dataset_root,
        split=split,
        image_size=image_size,
        max_objects=max_objects,
        train=train,
        max_samples=max_samples,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train if shuffle is None else shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_detection_batch,
    )


def create_joint_train_dataloader(
    primary_dataset_root: str,
    batch_size: int,
    image_size: int = 512,
    max_objects: int = 3,
    num_workers: int = 4,
    primary_max_samples: Optional[int] = None,
    coco_annotation_path: Optional[str] = None,
    coco_image_dir: Optional[str] = None,
    coco_max_samples: Optional[int] = None,
) -> DataLoader:
    datasets: List[Dataset] = [
        FixedSlotCocoDetectionDataset(
            dataset_root=primary_dataset_root,
            split="train",
            image_size=image_size,
            max_objects=max_objects,
            train=True,
            max_samples=primary_max_samples,
        )
    ]
    if coco_annotation_path and coco_image_dir:
        datasets.append(
            GenericCocoDetectionDataset(
                annotation_path=coco_annotation_path,
                image_dir=coco_image_dir,
                image_size=image_size,
                max_objects=max_objects,
                train=True,
                max_samples=coco_max_samples,
            )
        )

    dataset: Dataset
    if len(datasets) == 1:
        dataset = datasets[0]
    else:
        dataset = ConcatDataset(datasets)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_detection_batch,
    )


def prepare_inference_image(
    image: Image.Image,
    image_size: int = 512,
) -> Tuple[torch.Tensor, LetterboxMetadata]:
    canvas, _, meta = letterbox_image_and_boxes(
        image=image,
        boxes=torch.zeros((0, 4), dtype=torch.float32),
        image_size=image_size,
        scale_multiplier=1.0,
        random_offset=False,
    )
    image_tensor = TF.to_tensor(canvas)
    image_tensor = TF.normalize(image_tensor, IMAGENET_MEAN, IMAGENET_STD)
    return image_tensor, meta
