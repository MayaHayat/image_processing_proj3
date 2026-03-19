import json
import os
import time
from types import SimpleNamespace
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import draw_bounding_boxes, make_grid

from part2_dataset import IMAGENET_MEAN, IMAGENET_STD, create_dataloader
from part2_losses import calculate_iou
from part2_model import MobileNetDetector


def get_device() -> torch.device:
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DATASET_PATH = "./dog-2"
IMAGE_SIZE = 224
EPOCHS = 60
NUM_WORKERS = 0
EARLY_STOPPING_PATIENCE = 8
OUTPUT_DIR = os.path.join("outputs", "part2")
RUN_NAME = None
CONFIGS = [
    {"name": "LR_1e-3_BS_16", "lr": 0.001, "bs": 16}
    # {"name": "LR_1e-4_BS_16", "lr": 0.0001, "bs": 16},
    # {"name": "LR_5e-5_BS_16", "lr": 0.00005, "bs": 16},
    # {"name": "LR_1e-3_BS_32", "lr": 0.001, "bs": 32},
    # {"name": "LR_1e-4_BS_32", "lr": 0.0001, "bs": 32},
    # {"name": "LR_5e-5_BS_32", "lr": 0.00005, "bs": 32},
    # {"name": "LR_1e-3_BS_64", "lr": 0.001, "bs": 64},
    # {"name": "LR_1e-4_BS_64", "lr": 0.0001, "bs": 64},
    # {"name": "LR_5e-5_BS_64", "lr": 0.00005, "bs": 64},
]


def denormalize_image(image: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=image.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=image.device).view(3, 1, 1)
    return (image * std + mean).clamp(0.0, 1.0)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_run_args(cfg: Dict[str, float], device: torch.device, run_name: str) -> SimpleNamespace:
    return SimpleNamespace(
        dataset_root=os.path.abspath(DATASET_PATH),
        image_size=IMAGE_SIZE,
        epochs=EPOCHS,
        num_workers=NUM_WORKERS,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        output_dir=os.path.abspath(OUTPUT_DIR),
        run_name=run_name,
        config_name=cfg["name"],
        learning_rate=float(cfg["lr"]),
        batch_size=int(cfg["bs"]),
        device=str(device),
    )


def save_checkpoint(
    path: str,
    model: MobileNetDetector,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    args: SimpleNamespace,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": None,
            "metrics": metrics,
            "class_names": ["dog"],
            "args": vars(args),
        },
        path,
    )


def log_images_to_tensorboard(
    writer: SummaryWriter,
    epoch: int,
    images: torch.Tensor,
    targets: torch.Tensor,
    predictions: torch.Tensor,
    tag: str = "Val_Samples",
) -> None:
    preview_images = []
    num_to_log = min(4, images.shape[0])
    image_size = images.shape[-1]

    for sample_idx in range(num_to_log):
        image = (denormalize_image(images[sample_idx]).cpu() * 255).to(torch.uint8)
        pred_box = (predictions[sample_idx].detach().cpu().clamp(0.0, 1.0) * image_size).unsqueeze(0)
        gt_box = (targets[sample_idx].detach().cpu().clamp(0.0, 1.0) * image_size).unsqueeze(0)

        annotated = draw_bounding_boxes(
            image,
            boxes=gt_box,
            labels=["GT"],
            colors="green",
            width=3,
        )
        annotated = draw_bounding_boxes(
            annotated,
            boxes=pred_box,
            labels=["Pred"],
            colors="red",
            width=3,
        )
        preview_images.append(annotated.float() / 255.0)

    writer.add_image(tag, make_grid(preview_images, nrow=2), epoch)


def run_experiment(cfg: Dict[str, float], device: torch.device) -> Dict[str, float]:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = RUN_NAME or f"run_{cfg['name']}_{timestamp}"
    run_dir = os.path.join(OUTPUT_DIR, run_name)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    tensorboard_dir = os.path.join(run_dir, "tensorboard")
    ensure_dir(checkpoint_dir)
    ensure_dir(tensorboard_dir)
    run_args = build_run_args(cfg, device, run_name)

    print(f"\n>>> Starting Experiment: {cfg['name']} (outputs in {run_dir})")

    train_loader = create_dataloader(
        dataset_root=DATASET_PATH,
        split="train",
        batch_size=int(cfg["bs"]),
        image_size=IMAGE_SIZE,
        train=True,
        num_workers=NUM_WORKERS,
    )
    val_loader = create_dataloader(
        dataset_root=DATASET_PATH,
        split="valid",
        batch_size=int(cfg["bs"]),
        image_size=IMAGE_SIZE,
        train=False,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    model = MobileNetDetector().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))
    criterion = nn.SmoothL1Loss()
    writer = SummaryWriter(log_dir=tensorboard_dir)

    best_val_loss = float("inf")
    best_val_iou = 0.0
    counter = 0
    history: Dict[str, Dict[str, Dict[str, float]]] = {}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_iou = 0.0

        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_iou += calculate_iou(predictions, targets).item()

        avg_train_loss = train_loss / max(len(train_loader), 1)
        avg_train_iou = train_iou / max(len(train_loader), 1)

        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        with torch.no_grad():
            for batch_index, (images, targets) in enumerate(val_loader):
                images = images.to(device)
                targets = targets.to(device)
                predictions = model(images)
                val_loss += criterion(predictions, targets).item()
                val_iou += calculate_iou(predictions, targets).item()

                if batch_index == 0 and ((epoch - 1) % 5 == 0 or epoch == EPOCHS):
                    log_images_to_tensorboard(writer, epoch, images, targets, predictions)

        avg_val_loss = val_loss / max(len(val_loader), 1)
        avg_val_iou = val_iou / max(len(val_loader), 1)
        train_metrics = {
            "loss": avg_train_loss,
            "mean_iou": avg_train_iou,
        }
        valid_metrics = {
            "loss": avg_val_loss,
            "mean_iou": avg_val_iou,
        }
        history[str(epoch)] = {
            "train": train_metrics,
            "valid": valid_metrics,
            "lr": {"learning_rate": float(cfg["lr"])},
        }

        iou_improved = avg_val_iou > best_val_iou
        if iou_improved:
            best_val_iou = avg_val_iou

        writer.add_scalar("train/loss", avg_train_loss, epoch)
        writer.add_scalar("train/mean_iou", avg_train_iou, epoch)
        writer.add_scalar("valid/loss", avg_val_loss, epoch)
        writer.add_scalar("valid/mean_iou", avg_val_iou, epoch)
        writer.add_scalar("lr/learning_rate", float(cfg["lr"]), epoch)

        print(
            f"epoch={epoch:03d} train_loss={avg_train_loss:.4f} "
            f"valid_loss={avg_val_loss:.4f} valid_iou={avg_val_iou:.4f}"
        )

        last_checkpoint = os.path.join(checkpoint_dir, "last.pt")
        save_checkpoint(last_checkpoint, model, optimizer, epoch, valid_metrics, run_args)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(checkpoint_dir, "best.pt")
            save_checkpoint(save_path, model, optimizer, epoch, valid_metrics, run_args)
            print(f"--> Saved best model (by loss) to {save_path}")
            counter = 0
        else:
            counter += 1
            if counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    writer.close()
    summary = {
        "class_names": ["dog"],
        "best_valid_iou": best_val_iou,
        "best_valid_loss": best_val_loss,
        "test_metrics": {},
        "history": history,
        "dataset_sizes": {
            "train": len(train_loader.dataset),
            "valid": len(val_loader.dataset),
        },
        "args": vars(run_args),
    }
    with open(os.path.join(run_dir, "training_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return {
        "name": cfg["name"],
        "run_name": run_name,
        "best_valid_iou": best_val_iou,
        "best_valid_loss": best_val_loss,
    }


def main() -> None:
    device = get_device()
    print(f"Using device: {device}")

    ensure_dir(OUTPUT_DIR)
    if os.path.exists(OUTPUT_DIR):
        print("Success: Folder found!")
        print("Files currently inside:", os.listdir(OUTPUT_DIR))
    else:
        print("Error: Path not found. Check your folder names.")
        return

    config_results: List[Dict[str, float]] = []
    for cfg in CONFIGS:
        config_results.append(run_experiment(cfg, device))

    print("\n" + "=" * 60)
    print("BEST VALIDATION IoU PER CONFIG")
    print("=" * 60)
    for result in config_results:
        print(f"  {result['name']}: {result['best_valid_iou']:.4f} ({result['run_name']})")
    print("=" * 60)

    best_overall = max(config_results, key=lambda item: item["best_valid_iou"])
    print(
        f"Best overall: {best_overall['name']} "
        f"with Val IoU {best_overall['best_valid_iou']:.4f}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
