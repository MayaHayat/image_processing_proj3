import itertools
import json
import math
import os
import random
import time
from types import SimpleNamespace
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import draw_bounding_boxes, make_grid

from part3_dataset import CLASS_NAMES, create_dataloader, create_joint_train_dataloader
from part3_dataset import cxcywh_to_xyxy as dataset_cxcywh_to_xyxy
from part3_losses import (
    compute_average_precision,
    compute_detection_loss,
    exact_detection_matches,
    pairwise_iou,
    postprocess_detections,
    update_average_precision_records,
)
from part3_model import MobileNetV3FixedSlotDetector


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
DATASET_ROOT = "dogs-and-people-1"
IMAGE_SIZE = 512
MAX_OBJECTS = 3
BATCH_SIZE = 8
EPOCHS = 50
WARMUP_EPOCHS = 4
HEAD_LR = 1e-3
BACKBONE_LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0
DROPOUT = 0.2
SCORE_THRESHOLD = 0.45
NMS_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
GRADIENT_CLIP = 1.0
EARLY_STOPPING_PATIENCE = 8
SEED = 42
OUTPUT_DIR = os.path.join("outputs", "part3")
RUN_NAME = None
MAX_TRAIN_SAMPLES = None
MAX_VALID_SAMPLES = None
MAX_TEST_SAMPLES = None
USE_COCO = True
COCO_ANNOTATION_PATH = "custom_annotations.json"
COCO_IMAGE_DIR = "custom_coco_dataset"
COCO_MAX_SAMPLES = None
DEVICE = get_default_device()


def get_config() -> SimpleNamespace:
    return SimpleNamespace(
        dataset_root=resolve_project_path(DATASET_ROOT),
        image_size=IMAGE_SIZE,
        max_objects=MAX_OBJECTS,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        warmup_epochs=WARMUP_EPOCHS,
        head_lr=HEAD_LR,
        backbone_lr=BACKBONE_LR,
        weight_decay=WEIGHT_DECAY,
        num_workers=NUM_WORKERS,
        dropout=DROPOUT,
        score_threshold=SCORE_THRESHOLD,
        nms_threshold=NMS_THRESHOLD,
        iou_threshold=IOU_THRESHOLD,
        gradient_clip=GRADIENT_CLIP,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        seed=SEED,
        output_dir=resolve_project_path(OUTPUT_DIR),
        run_name=RUN_NAME,
        max_train_samples=MAX_TRAIN_SAMPLES,
        max_valid_samples=MAX_VALID_SAMPLES,
        max_test_samples=MAX_TEST_SAMPLES,
        use_coco=USE_COCO,
        coco_annotation_path=resolve_project_path(COCO_ANNOTATION_PATH),
        coco_image_dir=resolve_project_path(COCO_IMAGE_DIR),
        coco_max_samples=COCO_MAX_SAMPLES,
        device=DEVICE,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def resolve_optional_coco_paths(args: SimpleNamespace) -> Tuple[str, str] | Tuple[None, None]:
    if not args.use_coco:
        return None, None
    annotation_path = args.coco_annotation_path
    image_dir = args.coco_image_dir
    if not os.path.exists(annotation_path):
        raise FileNotFoundError(f"COCO annotation file not found: {annotation_path}")
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"COCO image directory not found: {image_dir}")
    return annotation_path, image_dir


def localization_matches(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    gt_boxes: torch.Tensor,
    iou_threshold: float,
) -> List[Tuple[int, int]]:
    if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
        return []

    iou_matrix = pairwise_iou(pred_boxes, gt_boxes)
    pred_count = pred_boxes.shape[0]
    gt_count = gt_boxes.shape[0]
    max_pairs = min(pred_count, gt_count)
    best_pairs: List[Tuple[int, int]] = []
    best_match_count = -1
    best_iou_sum = -1.0
    best_score_sum = -1.0

    for match_count in range(max_pairs + 1):
        for pred_perm in itertools.permutations(range(pred_count), match_count):
            for gt_perm in itertools.permutations(range(gt_count), match_count):
                valid = True
                iou_sum = 0.0
                score_sum = 0.0
                for pred_idx, gt_idx in zip(pred_perm, gt_perm):
                    iou_value = float(iou_matrix[pred_idx, gt_idx].item())
                    if iou_value < iou_threshold:
                        valid = False
                        break
                    iou_sum += iou_value
                    score_sum += float(pred_scores[pred_idx].item())
                if valid:
                    if (
                        match_count > best_match_count
                        or (match_count == best_match_count and iou_sum > best_iou_sum)
                        or (
                            match_count == best_match_count
                            and math.isclose(iou_sum, best_iou_sum)
                            and score_sum > best_score_sum
                        )
                    ):
                        best_match_count = match_count
                        best_iou_sum = iou_sum
                        best_score_sum = score_sum
                        best_pairs = list(zip(pred_perm, gt_perm))
    return best_pairs


def evaluate_batch_metrics(
    detections: Sequence[Dict[str, torch.Tensor]],
    targets: Dict[str, torch.Tensor],
    ap_records: List[List[Tuple[float, int]]],
    gt_counts: List[int],
    iou_threshold: float,
) -> Dict[str, float]:
    tp = 0
    fp = 0
    fn = 0
    matched_iou_sum = 0.0
    matched_iou_count = 0
    class_correct = 0
    class_total = 0

    update_average_precision_records(
        records=ap_records,
        gt_counts=gt_counts,
        detections=detections,
        gt_boxes_batch=targets["boxes"],
        gt_labels_batch=targets["labels"],
        object_mask_batch=targets["object_mask"],
        iou_threshold=iou_threshold,
    )

    for detection, gt_boxes, gt_labels, object_mask in zip(
        detections,
        targets["boxes"],
        targets["labels"],
        targets["object_mask"],
    ):
        valid_gt_boxes = dataset_cxcywh_to_xyxy(gt_boxes[object_mask.bool()])
        valid_gt_labels = gt_labels[object_mask.bool()]

        matches, unmatched_pred, unmatched_gt = exact_detection_matches(
            pred_boxes=detection["boxes"],
            pred_labels=detection["labels"],
            pred_scores=detection["scores"],
            gt_boxes=valid_gt_boxes,
            gt_labels=valid_gt_labels,
            iou_threshold=iou_threshold,
        )
        tp += len(matches)
        fp += len(unmatched_pred)
        fn += len(unmatched_gt)
        for pred_idx, gt_idx in matches:
            iou_value = float(pairwise_iou(
                detection["boxes"][pred_idx : pred_idx + 1],
                valid_gt_boxes[gt_idx : gt_idx + 1],
            )[0, 0].item())
            matched_iou_sum += iou_value
            matched_iou_count += 1

        localization_pairs = localization_matches(
            pred_boxes=detection["boxes"],
            pred_scores=detection["scores"],
            gt_boxes=valid_gt_boxes,
            iou_threshold=iou_threshold,
        )
        class_total += len(localization_pairs)
        for pred_idx, gt_idx in localization_pairs:
            class_correct += int(detection["labels"][pred_idx].item() == valid_gt_labels[gt_idx].item())

    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "matched_iou_sum": matched_iou_sum,
        "matched_iou_count": float(matched_iou_count),
        "class_correct": float(class_correct),
        "class_total": float(class_total),
    }


def build_optimizer(model: MobileNetV3FixedSlotDetector, args: SimpleNamespace) -> AdamW:
    return AdamW(
        [
            {"params": list(model.backbone_parameters()), "lr": args.backbone_lr},
            {"params": list(model.head_parameters()), "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )


def build_scheduler(
    optimizer: AdamW,
    epochs: int,
    warmup_epochs: int,
) -> SequentialLR:
    warmup_epochs = max(1, warmup_epochs)
    cosine_epochs = max(1, epochs - warmup_epochs)
    warmup = LinearLR(optimizer, start_factor=0.2, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=cosine_epochs)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])


def move_targets_to_device(targets: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in targets.items()}


def run_epoch(
    model: MobileNetV3FixedSlotDetector,
    loader,
    optimizer: AdamW,
    device: torch.device,
    training: bool,
    epoch_index: int,
    args: SimpleNamespace,
) -> Dict[str, float]:
    mode_name = "train" if training else "valid"
    model.train(training)

    ap_records: List[List[Tuple[float, int]]] = [[] for _ in CLASS_NAMES]
    gt_counts = [0 for _ in CLASS_NAMES]
    running = {
        "loss": 0.0,
        "box_loss": 0.0,
        "iou_loss": 0.0,
        "objectness_loss": 0.0,
        "class_loss": 0.0,
        "matched_iou": 0.0,
        "tp": 0.0,
        "fp": 0.0,
        "fn": 0.0,
        "matched_iou_sum": 0.0,
        "matched_iou_count": 0.0,
        "class_correct": 0.0,
        "class_total": 0.0,
        "images": 0.0,
    }

    for batch_index, (images, targets) in enumerate(loader):
        images = images.to(device)
        targets = move_targets_to_device(targets, device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            outputs = model(images)
            loss_dict = compute_detection_loss(outputs, targets)
            if training:
                loss_dict["loss"].backward()
                clip_grad_norm_(model.parameters(), max_norm=args.gradient_clip)
                optimizer.step()

        detections = postprocess_detections(
            outputs=outputs,
            score_threshold=args.score_threshold,
            nms_threshold=args.nms_threshold,
        )
        batch_stats = evaluate_batch_metrics(
            detections=detections,
            targets=targets,
            ap_records=ap_records,
            gt_counts=gt_counts,
            iou_threshold=args.iou_threshold,
        )

        running["loss"] += float(loss_dict["loss"].item())
        running["box_loss"] += float(loss_dict["box_loss"].item())
        running["iou_loss"] += float(loss_dict["iou_loss"].item())
        running["objectness_loss"] += float(loss_dict["objectness_loss"].item())
        running["class_loss"] += float(loss_dict["class_loss"].item())
        running["matched_iou"] += float(loss_dict["matched_iou"].item())
        running["images"] += float(images.shape[0])
        for key in batch_stats:
            running[key] += batch_stats[key]

        if batch_index % 20 == 0:
            print(
                f"[{mode_name}] epoch={epoch_index:03d} batch={batch_index:04d}/{len(loader):04d} "
                f"loss={loss_dict['loss'].item():.4f}"
            )

    batches = max(len(loader), 1)
    precision = running["tp"] / max(running["tp"] + running["fp"], 1.0)
    recall = running["tp"] / max(running["tp"] + running["fn"], 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-6)
    mean_iou = running["matched_iou_sum"] / max(running["matched_iou_count"], 1.0)
    class_accuracy = running["class_correct"] / max(running["class_total"], 1.0)
    false_positives_per_image = running["fp"] / max(running["images"], 1.0)
    ap_values = {
        f"ap50_{class_name}": compute_average_precision(ap_records[class_idx], gt_counts[class_idx])
        for class_idx, class_name in enumerate(CLASS_NAMES)
    }
    map50 = sum(ap_values.values()) / max(len(ap_values), 1)

    metrics = {
        "loss": running["loss"] / batches,
        "box_loss": running["box_loss"] / batches,
        "iou_loss": running["iou_loss"] / batches,
        "objectness_loss": running["objectness_loss"] / batches,
        "class_loss": running["class_loss"] / batches,
        "matched_iou_loss_proxy": running["matched_iou"] / batches,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": mean_iou,
        "class_accuracy": class_accuracy,
        "false_positives_per_image": false_positives_per_image,
        "map50": map50,
    }
    metrics.update(ap_values)
    return metrics


def log_metrics(writer: SummaryWriter, prefix: str, metrics: Dict[str, float], epoch: int) -> None:
    for key, value in metrics.items():
        writer.add_scalar(f"{prefix}/{key}", value, epoch)


def denormalize_image(image: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(3, 1, 1)
    return (image * std + mean).clamp(0.0, 1.0)


def build_visualization_grid(
    model: MobileNetV3FixedSlotDetector,
    loader,
    device: torch.device,
    score_threshold: float,
    nms_threshold: float,
) -> torch.Tensor:
    model_was_training = model.training
    model.eval()
    images, targets = next(iter(loader))
    images = images.to(device)
    targets = move_targets_to_device(targets, device)
    with torch.no_grad():
        outputs = model(images)
        detections = postprocess_detections(
            outputs=outputs,
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
        )
    if model_was_training:
        model.train()

    preview_images = []
    max_images = min(4, images.shape[0])
    for sample_idx in range(max_images):
        image = (denormalize_image(images[sample_idx]).cpu() * 255).to(torch.uint8)
        gt_mask = targets["object_mask"][sample_idx].bool()
        gt_boxes = dataset_cxcywh_to_xyxy(targets["boxes"][sample_idx][gt_mask]).cpu()
        gt_labels = targets["labels"][sample_idx][gt_mask].cpu()
        pred_boxes = detections[sample_idx]["boxes"].cpu()
        pred_labels = detections[sample_idx]["labels"].cpu()
        pred_scores = detections[sample_idx]["scores"].cpu()

        annotated = image
        if gt_boxes.numel() > 0:
            gt_text = [f"GT:{CLASS_NAMES[label.item()]}" for label in gt_labels]
            annotated = draw_bounding_boxes(
                annotated,
                boxes=gt_boxes * image.shape[1],
                labels=gt_text,
                colors="green",
                width=3,
            )
        if pred_boxes.numel() > 0:
            pred_text = [
                f"P:{CLASS_NAMES[label.item()]} {score:.2f}"
                for label, score in zip(pred_labels, pred_scores)
            ]
            annotated = draw_bounding_boxes(
                annotated,
                boxes=pred_boxes * image.shape[1],
                labels=pred_text,
                colors="red",
                width=3,
            )
        preview_images.append(annotated.float() / 255.0)
    return make_grid(preview_images, nrow=2)


def save_checkpoint(
    path: str,
    model: MobileNetV3FixedSlotDetector,
    optimizer: AdamW,
    scheduler: SequentialLR,
    epoch: int,
    metrics: Dict[str, float],
    args: SimpleNamespace,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "metrics": metrics,
            "class_names": CLASS_NAMES,
            "args": vars(args),
        },
        path,
    )


def main() -> None:
    args = get_config()
    set_seed(args.seed)

    run_name = args.run_name or time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, run_name)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    tensorboard_dir = os.path.join(run_dir, "tensorboard")
    ensure_dir(checkpoint_dir)
    ensure_dir(tensorboard_dir)

    device = torch.device(args.device)
    coco_annotation_path, coco_image_dir = resolve_optional_coco_paths(args)
    train_loader = create_joint_train_dataloader(
        primary_dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        image_size=args.image_size,
        max_objects=args.max_objects,
        num_workers=args.num_workers,
        primary_max_samples=args.max_train_samples,
        coco_annotation_path=coco_annotation_path,
        coco_image_dir=coco_image_dir,
        coco_max_samples=args.coco_max_samples,
    )
    valid_loader = create_dataloader(
        dataset_root=args.dataset_root,
        split="valid",
        batch_size=args.batch_size,
        image_size=args.image_size,
        max_objects=args.max_objects,
        train=False,
        num_workers=args.num_workers,
        shuffle=False,
        max_samples=args.max_valid_samples,
    )
    test_loader = create_dataloader(
        dataset_root=args.dataset_root,
        split="test",
        batch_size=args.batch_size,
        image_size=args.image_size,
        max_objects=args.max_objects,
        train=False,
        num_workers=args.num_workers,
        shuffle=False,
        max_samples=args.max_test_samples,
    )

    model = MobileNetV3FixedSlotDetector(
        num_classes=len(CLASS_NAMES),
        num_slots=args.max_objects,
        dropout=args.dropout,
        pretrained=True,
    ).to(device)
    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args.epochs, args.warmup_epochs)
    writer = SummaryWriter(log_dir=tensorboard_dir)

    best_map50 = -1.0
    epochs_without_improvement = 0
    history: Dict[str, Dict[str, Dict[str, float]]] = {}

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, True, epoch, args)
        valid_metrics = run_epoch(model, valid_loader, optimizer, device, False, epoch, args)
        scheduler.step()

        current_lrs = {
            "backbone_lr": optimizer.param_groups[0]["lr"],
            "head_lr": optimizer.param_groups[1]["lr"],
        }
        writer.add_scalar("lr/backbone", current_lrs["backbone_lr"], epoch)
        writer.add_scalar("lr/head", current_lrs["head_lr"], epoch)
        log_metrics(writer, "train", train_metrics, epoch)
        log_metrics(writer, "valid", valid_metrics, epoch)
        writer.add_image(
            "valid/predictions_vs_ground_truth",
            build_visualization_grid(
                model=model,
                loader=valid_loader,
                device=device,
                score_threshold=args.score_threshold,
                nms_threshold=args.nms_threshold,
            ),
            epoch,
        )

        history[str(epoch)] = {
            "train": train_metrics,
            "valid": valid_metrics,
            "lr": current_lrs,
        }
        print(
            f"epoch={epoch:03d} train_loss={train_metrics['loss']:.4f} "
            f"valid_loss={valid_metrics['loss']:.4f} "
            f"valid_iou={valid_metrics['mean_iou']:.4f} "
            f"valid_clsacc={valid_metrics['class_accuracy']:.4f} "
            f"valid_map50={valid_metrics['map50']:.4f}"
        )

        last_checkpoint = os.path.join(checkpoint_dir, "last.pt")
        save_checkpoint(last_checkpoint, model, optimizer, scheduler, epoch, valid_metrics, args)
        if valid_metrics["map50"] > best_map50:
            best_map50 = valid_metrics["map50"]
            epochs_without_improvement = 0
            best_checkpoint = os.path.join(checkpoint_dir, "best.pt")
            save_checkpoint(best_checkpoint, model, optimizer, scheduler, epoch, valid_metrics, args)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.early_stopping_patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    best_checkpoint_path = os.path.join(checkpoint_dir, "best.pt")
    if os.path.exists(best_checkpoint_path):
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        test_metrics = run_epoch(model, test_loader, optimizer, device, False, 0, args)
    else:
        test_metrics = {}

    writer.close()
    summary = {
        "class_names": CLASS_NAMES,
        "best_valid_map50": best_map50,
        "test_metrics": test_metrics,
        "history": history,
        "dataset_sizes": {
            "train": len(train_loader.dataset),
            "valid": len(valid_loader.dataset),
            "test": len(test_loader.dataset),
        },
        "used_coco": bool(args.use_coco),
        "args": vars(args),
    }
    with open(os.path.join(run_dir, "training_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary["test_metrics"], indent=2))


if __name__ == "__main__":
    main()
