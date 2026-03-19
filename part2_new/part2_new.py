import json
import os
import random
import time
import xml.etree.ElementTree as ET
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torchvision.utils as vutils
from PIL import Image
from roboflow import Roboflow
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import models, transforms
from torchvision.models import MobileNet_V3_Large_Weights


def get_project_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def resolve_project_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(get_project_root(), path)


def get_default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


PROJECT_ROOT = get_project_root()

# Update these values directly before running the script.
DATASET_ROOT = "dog-2"
OUTPUT_DIR = os.path.join("outputs", "part2")
DATASET_REPORT_NAME = "dataset_dimensions_report.csv"
IMAGE_SIZE = 224
EPOCHS = 60
NUM_WORKERS = 0
EARLY_STOPPING_PATIENCE = 8
SEED = 42
DEVICE = get_default_device()
RUN_NAME = None

ROBOFLOW_API_KEY = "zcApQr54bnD6POqEKFmP"
ROBOFLOW_WORKSPACE = "maya-mlots"
ROBOFLOW_PROJECT = "dog-nmpmi-f1sao"
ROBOFLOW_VERSION = 2
DOWNLOAD_DATASET = True

RUN_DATASET_ANALYSIS = True

EXPERIMENT_CONFIGS = [
    {"name": "LR_1e-3_BS_16", "lr": 0.001, "bs": 16},
    {"name": "LR_1e-4_BS_16", "lr": 0.0001, "bs": 16},
    {"name": "LR_5e-5_BS_16", "lr": 0.00005, "bs": 16},
    {"name": "LR_1e-3_BS_32", "lr": 0.001, "bs": 32},
    {"name": "LR_1e-4_BS_32", "lr": 0.0001, "bs": 32},
    {"name": "LR_5e-5_BS_32", "lr": 0.00005, "bs": 32},
    {"name": "LR_1e-3_BS_64", "lr": 0.001, "bs": 64},
    {"name": "LR_1e-4_BS_64", "lr": 0.0001, "bs": 64},
    {"name": "LR_5e-5_BS_64", "lr": 0.00005, "bs": 64},
]


def get_config() -> SimpleNamespace:
    return SimpleNamespace(
        dataset_root=resolve_project_path(DATASET_ROOT),
        output_dir=resolve_project_path(OUTPUT_DIR),
        dataset_report_path=resolve_project_path(DATASET_REPORT_NAME),
        image_size=IMAGE_SIZE,
        epochs=EPOCHS,
        num_workers=NUM_WORKERS,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        seed=SEED,
        device=DEVICE,
        run_name=RUN_NAME,
        roboflow_api_key=ROBOFLOW_API_KEY,
        roboflow_workspace=ROBOFLOW_WORKSPACE,
        roboflow_project=ROBOFLOW_PROJECT,
        roboflow_version=ROBOFLOW_VERSION,
        download_dataset=DOWNLOAD_DATASET,
        run_dataset_analysis=RUN_DATASET_ANALYSIS,
        experiment_configs=EXPERIMENT_CONFIGS,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_train_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def download_dataset_if_needed(args: SimpleNamespace) -> None:
    if not args.download_dataset:
        return

    rf = Roboflow(api_key=args.roboflow_api_key)
    project = rf.workspace(args.roboflow_workspace).project(args.roboflow_project)
    version = project.version(args.roboflow_version)
    version.download("voc")
    print(f"\nTotal images in dataset: {version.images}")


def get_dimensions_df(root_path: str) -> pd.DataFrame:
    data = []
    for split in ["train", "valid", "test"]:
        split_path = os.path.join(root_path, split)
        if not os.path.isdir(split_path):
            continue

        xml_files = sorted(file_name for file_name in os.listdir(split_path) if file_name.endswith(".xml"))
        for file_name in xml_files:
            xml_path = os.path.join(split_path, file_name)
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find("size")
            if size is None:
                continue

            data.append(
                {
                    "split": split,
                    "filename": file_name.replace(".xml", ".jpg"),
                    "width": int(size.find("width").text),
                    "height": int(size.find("height").text),
                }
            )

    return pd.DataFrame(data)


def get_full_dataset_df(root_path: str) -> pd.DataFrame:
    data = []
    for split in ["train", "valid", "test"]:
        split_path = os.path.join(root_path, split)
        if not os.path.isdir(split_path):
            continue

        xml_files = sorted(file_name for file_name in os.listdir(split_path) if file_name.endswith(".xml"))
        for file_name in xml_files:
            tree = ET.parse(os.path.join(split_path, file_name))
            root = tree.getroot()
            size = root.find("size")
            width = int(size.find("width").text)
            height = int(size.find("height").text)

            for obj in root.findall("object"):
                box = obj.find("bndbox")
                data.append(
                    {
                        "split": split,
                        "filename": file_name.replace(".xml", ".jpg"),
                        "width": width,
                        "height": height,
                        "class": obj.find("name").text,
                        "xmin": int(box.find("xmin").text),
                        "ymin": int(box.find("ymin").text),
                        "xmax": int(box.find("xmax").text),
                        "ymax": int(box.find("ymax").text),
                    }
                )

    return pd.DataFrame(data)


def save_dataset_report(root_path: str, report_path: str) -> pd.DataFrame:
    df_dims = get_dimensions_df(root_path)
    print(f"Processed {len(df_dims)} XML files.")
    print("\n--- Summary Statistics ---")
    print(df_dims[["width", "height"]].describe().loc[["min", "max", "mean"]])
    df_dims.to_csv(report_path, index=False)
    print(f"\nResults saved to '{report_path}'.")
    return df_dims


def visualize_from_df(df: pd.DataFrame, root_path: str, num_samples: int = 8) -> None:
    if df.empty:
        print("Dataset visualization skipped because no annotations were found.")
        return

    plt.figure(figsize=(20, 10))
    unique_files = df["filename"].unique()
    sample_files = random.sample(list(unique_files), min(num_samples, len(unique_files)))

    for index, filename in enumerate(sample_files):
        img_info = df[df["filename"] == filename].iloc[0]
        image_path = os.path.join(root_path, img_info["split"], filename)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes = df[df["filename"] == filename]
        for _, box in boxes.iterrows():
            cv2.rectangle(image, (box["xmin"], box["ymin"]), (box["xmax"], box["ymax"]), (0, 255, 0), 3)
            cv2.putText(
                image,
                box["class"],
                (box["xmin"], box["ymin"] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

        plt.subplot(2, 4, index + 1)
        plt.imshow(image)
        plt.title(f"{filename}\n{img_info['width']}x{img_info['height']}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


class Part2Dataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        augment: bool = False,
    ) -> None:
        self.root = os.path.join(root_dir, split)
        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"Directory not found: {self.root}")
        self.img_files = sorted(file_name for file_name in os.listdir(self.root) if file_name.endswith(".jpg"))
        self.transform = transform
        self.augment = augment

    def __len__(self) -> int:
        return len(self.img_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = os.path.join(self.root, self.img_files[idx])
        xml_path = img_path.replace(".jpg", ".xml")

        image = Image.open(img_path).convert("RGB")
        width, height = image.size

        tree = ET.parse(xml_path)
        root = tree.getroot()
        box = root.find("object").find("bndbox")

        xmin = float(box.find("xmin").text) / width
        ymin = float(box.find("ymin").text) / height
        xmax = float(box.find("xmax").text) / width
        ymax = float(box.find("ymax").text) / height
        target = torch.tensor([xmin, ymin, xmax, ymax], dtype=torch.float32)

        if self.augment:
            if random.random() < 0.5:
                image = TF.hflip(image)
                target = torch.tensor(
                    [1 - target[2], target[1], 1 - target[0], target[3]],
                    dtype=torch.float32,
                )
            if random.random() < 0.5:
                image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
            if random.random() < 0.5:
                image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))
            if random.random() < 0.5:
                image = TF.adjust_saturation(image, random.uniform(0.8, 1.2))

        if self.transform is not None:
            image = self.transform(image)
        return image, target


class MobileNetDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        backbone = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
        self.features = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.regressor = nn.Sequential(
            nn.Linear(960, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.regressor(x)


def calculate_iou(preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    x1 = torch.max(preds[:, 0], labels[:, 0])
    y1 = torch.max(preds[:, 1], labels[:, 1])
    x2 = torch.min(preds[:, 2], labels[:, 2])
    y2 = torch.min(preds[:, 3], labels[:, 3])
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    area_pred = (preds[:, 2] - preds[:, 0]) * (preds[:, 3] - preds[:, 1])
    area_gt = (labels[:, 2] - labels[:, 0]) * (labels[:, 3] - labels[:, 1])
    union = area_pred + area_gt - intersection + 1e-6
    return (intersection / union).mean()


def denormalize_image(image: torch.Tensor) -> np.ndarray:
    image_np = image.cpu().permute(1, 2, 0).numpy()
    image_np = image_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
    return (image_np.clip(0, 1) * 255).astype("uint8")


def log_images_to_tb(
    writer: SummaryWriter,
    epoch: int,
    images: torch.Tensor,
    targets: torch.Tensor,
    preds: torch.Tensor,
    name: str = "Val_Samples",
) -> None:
    image_list = []
    num_to_log = min(4, images.shape[0])

    for index in range(num_to_log):
        image = denormalize_image(images[index]).copy()
        height, width, _ = image.shape

        pred_box = preds[index].detach().cpu().numpy()
        cv2.rectangle(
            image,
            (int(pred_box[0] * width), int(pred_box[1] * height)),
            (int(pred_box[2] * width), int(pred_box[3] * height)),
            (255, 0, 0),
            2,
        )

        target_box = targets[index].detach().cpu().numpy()
        cv2.rectangle(
            image,
            (int(target_box[0] * width), int(target_box[1] * height)),
            (int(target_box[2] * width), int(target_box[3] * height)),
            (0, 255, 0),
            2,
        )

        image_list.append(torch.from_numpy(image).permute(2, 0, 1))

    if image_list:
        grid = vutils.make_grid(image_list)
        writer.add_image(name, grid, epoch)


def build_dataloaders(
    dataset_root: str,
    batch_size: int,
    image_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        Part2Dataset(dataset_root, "train", build_train_transform(image_size), augment=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    valid_loader = DataLoader(
        Part2Dataset(dataset_root, "valid", build_eval_transform(image_size)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, valid_loader


def build_test_loader(
    dataset_root: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        Part2Dataset(dataset_root, "test", build_eval_transform(image_size)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    criterion: nn.Module,
    device: torch.device,
    training: bool,
) -> Dict[str, object]:
    model.train(training)
    running_loss = 0.0
    running_iou = 0.0
    preview_batch = None

    for batch_index, (images, targets) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            preds = model(images)
            loss = criterion(preds, targets)
            if training:
                loss.backward()
                optimizer.step()

        running_loss += float(loss.item())
        running_iou += float(calculate_iou(preds, targets).item())

        if not training and batch_index == 0:
            preview_batch = (
                images.detach().cpu(),
                targets.detach().cpu(),
                preds.detach().cpu(),
            )

    num_batches = max(len(loader), 1)
    return {
        "loss": running_loss / num_batches,
        "iou": running_iou / num_batches,
        "preview_batch": preview_batch,
    }


def log_metrics(writer: SummaryWriter, prefix: str, metrics: Dict[str, object], epoch: int) -> None:
    for key, value in metrics.items():
        if key == "preview_batch":
            continue
        writer.add_scalar(f"{prefix}/{key}", float(value), epoch)


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, object],
    args: SimpleNamespace,
    config: Dict[str, float],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metrics": {key: value for key, value in metrics.items() if key != "preview_batch"},
            "config": dict(config),
            "args": vars(args),
        },
        path,
    )


def train_single_config(
    args: SimpleNamespace,
    config: Dict[str, float],
    device: torch.device,
    run_dir: str,
) -> Dict[str, object]:
    config_dir = os.path.join(run_dir, config["name"])
    checkpoint_dir = os.path.join(config_dir, "checkpoints")
    tensorboard_dir = os.path.join(config_dir, "tensorboard")
    ensure_dir(config_dir)
    ensure_dir(checkpoint_dir)
    ensure_dir(tensorboard_dir)

    print(f"\n>>> Starting Experiment: {config['name']} (outputs in {config_dir})")
    train_loader, valid_loader = build_dataloaders(
        dataset_root=args.dataset_root,
        batch_size=config["bs"],
        image_size=args.image_size,
        num_workers=args.num_workers,
    )
    test_loader = build_test_loader(
        dataset_root=args.dataset_root,
        image_size=args.image_size,
        batch_size=config["bs"],
        num_workers=args.num_workers,
    )

    model = MobileNetDetector().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.SmoothL1Loss()
    writer = SummaryWriter(log_dir=tensorboard_dir)

    best_val_loss = float("inf")
    best_val_iou = 0.0
    epochs_without_improvement = 0
    history: Dict[str, Dict[str, float]] = {}

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, criterion, device, True)
        valid_metrics = run_epoch(model, valid_loader, None, criterion, device, False)

        writer.add_scalar("lr/main", optimizer.param_groups[0]["lr"], epoch)
        log_metrics(writer, "train", train_metrics, epoch)
        log_metrics(writer, "valid", valid_metrics, epoch)

        preview_batch = valid_metrics["preview_batch"]
        if preview_batch is not None and (epoch % 5 == 0 or epoch == args.epochs):
            preview_images, preview_targets, preview_preds = preview_batch
            log_images_to_tb(
                writer,
                epoch,
                preview_images,
                preview_targets,
                preview_preds,
            )

        history[str(epoch)] = {
            "train_loss": float(train_metrics["loss"]),
            "train_iou": float(train_metrics["iou"]),
            "val_loss": float(valid_metrics["loss"]),
            "val_iou": float(valid_metrics["iou"]),
        }
        print(
            f"epoch={epoch:03d} train_loss={train_metrics['loss']:.4f} "
            f"valid_loss={valid_metrics['loss']:.4f} "
            f"valid_iou={valid_metrics['iou']:.4f}"
        )

        last_checkpoint = os.path.join(checkpoint_dir, "last.pt")
        save_checkpoint(last_checkpoint, model, optimizer, epoch, valid_metrics, args, config)

        iou_improved = valid_metrics["iou"] > best_val_iou
        if iou_improved:
            best_val_iou = float(valid_metrics["iou"])

        if valid_metrics["loss"] < best_val_loss:
            best_val_loss = float(valid_metrics["loss"])
            save_path = os.path.join(checkpoint_dir, "best.pt")
            save_checkpoint(save_path, model, optimizer, epoch, valid_metrics, args, config)
            print(f"--> Saved best model (by loss) to {save_path}")
            epochs_without_improvement = 0
        elif iou_improved:
            save_path = os.path.join(checkpoint_dir, "best_iou.pt")
            save_checkpoint(save_path, model, optimizer, epoch, valid_metrics, args, config)
            print(f"--> Saved best model (by IoU) to {save_path}")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stopping_patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    best_checkpoint_path = os.path.join(checkpoint_dir, "best.pt")
    if os.path.exists(best_checkpoint_path):
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        test_metrics = run_epoch(model, test_loader, None, criterion, device, False)
    else:
        test_metrics = {}

    writer.close()
    config_summary_path = os.path.join(config_dir, "training_summary.json")
    with open(config_summary_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "config": dict(config),
                "best_val_loss": best_val_loss,
                "best_val_iou": best_val_iou,
                "test_metrics": {key: value for key, value in test_metrics.items() if key != "preview_batch"},
                "history": history,
                "dataset_sizes": {
                    "train": len(train_loader.dataset),
                    "valid": len(valid_loader.dataset),
                    "test": len(test_loader.dataset),
                },
                "args": vars(args),
            },
            handle,
            indent=2,
        )
    return {
        "name": config["name"],
        "lr": config["lr"],
        "batch_size": config["bs"],
        "best_val_loss": best_val_loss,
        "best_val_iou": best_val_iou,
        "test_metrics": {key: value for key, value in test_metrics.items() if key != "preview_batch"},
        "history": history,
        "checkpoint_dir": checkpoint_dir,
        "summary_path": config_summary_path,
    }


def print_training_summary(config_results: List[Dict[str, object]]) -> Dict[str, object]:
    print("\n" + "=" * 60)
    print("BEST VALIDATION IoU PER CONFIG")
    print("=" * 60)
    for result in config_results:
        print(f"  {result['name']}: {result['best_val_iou']:.4f}")
    print("=" * 60)
    best_overall = max(config_results, key=lambda item: item["best_val_iou"])
    print(
        f"Best overall: {best_overall['name']} "
        f"with Val IoU {best_overall['best_val_iou']:.4f}"
    )
    print("=" * 60)
    return best_overall


def load_model_for_evaluation(model_path: str, device: torch.device) -> MobileNetDetector:
    model = MobileNetDetector().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def plot_test_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    image_size: int,
    num_images: int = 16,
) -> None:
    model.eval()
    images, targets = next(iter(loader))
    images = images.to(device)
    targets = targets.to(device)

    with torch.no_grad():
        predictions = model(images)

    rows = 4
    max_images = min(num_images, images.shape[0])
    plt.figure(figsize=(18, 6 * rows))

    for index in range(max_images):
        image = images[index].cpu().permute(1, 2, 0).numpy()
        image = image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
        image = image.clip(0, 1)

        plt.subplot(rows, 4, index + 1)
        plt.imshow(image)

        pred_box = predictions[index].detach().cpu().numpy() * image_size
        gt_box = targets[index].detach().cpu().numpy() * image_size

        rect_pred = patches.Rectangle(
            (pred_box[0], pred_box[1]),
            pred_box[2] - pred_box[0],
            pred_box[3] - pred_box[1],
            linewidth=3,
            edgecolor="red",
            facecolor="none",
            label="Pred",
        )
        rect_gt = patches.Rectangle(
            (gt_box[0], gt_box[1]),
            gt_box[2] - gt_box[0],
            gt_box[3] - gt_box[1],
            linewidth=2,
            edgecolor="lime",
            facecolor="none",
            linestyle="--",
            label="GT",
        )

        ax = plt.gca()
        ax.add_patch(rect_pred)
        ax.add_patch(rect_gt)

        iou_score = calculate_iou(predictions[index : index + 1], targets[index : index + 1]).item()
        plt.title(f"Test Image {index + 1} | IoU: {iou_score:.2f}")
        plt.axis("off")

    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

def save_training_summary(
    run_dir: str,
    args: SimpleNamespace,
    config_results: List[Dict[str, object]],
    best_overall: Dict[str, object],
) -> None:
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
        "best_overall": {
            "name": best_overall["name"],
            "best_val_iou": best_overall["best_val_iou"],
            "best_val_loss": best_overall["best_val_loss"],
        },
        "config_results": [
            {
                "name": result["name"],
                "lr": result["lr"],
                "batch_size": result["batch_size"],
                "best_val_loss": result["best_val_loss"],
                "best_val_iou": result["best_val_iou"],
                "test_metrics": result["test_metrics"],
                "history": result["history"],
            }
            for result in config_results
        ],
    }
    summary_path = os.path.join(run_dir, "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Training summary saved to {summary_path}")


def main() -> None:
    args = get_config()
    set_seed(args.seed)
    ensure_dir(args.output_dir)
    run_name = args.run_name or time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, run_name)
    ensure_dir(run_dir)

    device = torch.device(args.device)
    print(f"Using device: {device}")
    if os.path.exists(args.output_dir):
        print("Success: Folder found!")
        print("Files currently inside:", os.listdir(args.output_dir))

    download_dataset_if_needed(args)

    if args.run_dataset_analysis:
        save_dataset_report(args.dataset_root, args.dataset_report_path)

    config_results = []
    for config in args.experiment_configs:
        config_results.append(train_single_config(args, config, device, run_dir))

    best_overall = print_training_summary(config_results)
    save_training_summary(run_dir, args, config_results, best_overall)


if __name__ == "__main__":
    main()
