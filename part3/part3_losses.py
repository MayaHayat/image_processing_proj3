import itertools
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torchvision.ops import nms

from part3_dataset import cxcywh_to_xyxy


def pairwise_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=boxes1.dtype, device=boxes1.device)

    top_left = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    inter_sizes = (bottom_right - top_left).clamp(min=0.0)
    inter_area = inter_sizes[..., 0] * inter_sizes[..., 1]

    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(min=0.0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0.0))[:, None]
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp(min=0.0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0.0))[None, :]
    union = area1 + area2 - inter_area
    return inter_area / union.clamp(min=1e-6)


def match_predictions(
    pred_boxes: torch.Tensor,
    pred_class_logits: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    class_cost_weight: float = 1.0,
    l1_cost_weight: float = 2.0,
    iou_cost_weight: float = 2.0,
) -> List[Tuple[int, int]]:
    num_gt = gt_boxes.shape[0]
    num_slots = pred_boxes.shape[0]
    if num_gt == 0:
        return []

    pred_xyxy = cxcywh_to_xyxy(pred_boxes)
    gt_xyxy = cxcywh_to_xyxy(gt_boxes)
    iou_matrix = pairwise_iou(pred_xyxy, gt_xyxy)
    class_probs = pred_class_logits.softmax(dim=-1)

    class_cost = -class_probs[:, gt_labels]
    l1_cost = torch.cdist(pred_boxes, gt_boxes, p=1)
    iou_cost = 1.0 - iou_matrix
    total_cost = (
        class_cost_weight * class_cost
        + l1_cost_weight * l1_cost
        + iou_cost_weight * iou_cost
    )

    best_pairs: List[Tuple[int, int]] = []
    best_cost = None
    for slot_perm in itertools.permutations(range(num_slots), num_gt):
        current_cost = 0.0
        pairs: List[Tuple[int, int]] = []
        for gt_idx, pred_idx in enumerate(slot_perm):
            current_cost += float(total_cost[pred_idx, gt_idx].item())
            pairs.append((pred_idx, gt_idx))
        if best_cost is None or current_cost < best_cost:
            best_cost = current_cost
            best_pairs = pairs
    return best_pairs


def compute_detection_loss(
    outputs: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    box_weight: float = 5.0,
    iou_weight: float = 2.0,
    objectness_weight: float = 1.0,
    class_weight: float = 1.0,
) -> Dict[str, torch.Tensor]:
    pred_boxes = outputs["boxes"]
    pred_objectness = outputs["objectness_logits"]
    pred_class_logits = outputs["class_logits"]

    target_boxes = targets["boxes"]
    target_labels = targets["labels"]
    object_mask = targets["object_mask"].bool()

    batch_size, num_slots = pred_boxes.shape[:2]
    total_box = pred_boxes.new_tensor(0.0)
    total_iou = pred_boxes.new_tensor(0.0)
    total_objectness = pred_boxes.new_tensor(0.0)
    total_class = pred_boxes.new_tensor(0.0)
    matched_ious: List[torch.Tensor] = []

    for batch_idx in range(batch_size):
        gt_boxes = target_boxes[batch_idx][object_mask[batch_idx]]
        gt_labels = target_labels[batch_idx][object_mask[batch_idx]]
        obj_targets = pred_boxes.new_zeros(num_slots)

        if gt_boxes.shape[0] > 0:
            matches = match_predictions(
                pred_boxes=pred_boxes[batch_idx],
                pred_class_logits=pred_class_logits[batch_idx],
                gt_boxes=gt_boxes,
                gt_labels=gt_labels,
            )
            pred_indices = torch.tensor([pred_idx for pred_idx, _ in matches], device=pred_boxes.device, dtype=torch.long)
            gt_indices = torch.tensor([gt_idx for _, gt_idx in matches], device=pred_boxes.device, dtype=torch.long)
            obj_targets[pred_indices] = 1.0

            matched_pred_boxes = pred_boxes[batch_idx, pred_indices]
            matched_gt_boxes = gt_boxes[gt_indices]
            matched_pred_xyxy = cxcywh_to_xyxy(matched_pred_boxes)
            matched_gt_xyxy = cxcywh_to_xyxy(matched_gt_boxes)
            matched_iou = pairwise_iou(matched_pred_xyxy, matched_gt_xyxy).diag()

            total_box = total_box + F.smooth_l1_loss(
                matched_pred_boxes,
                matched_gt_boxes,
                reduction="mean",
                beta=0.1,
            )
            total_iou = total_iou + (1.0 - matched_iou).mean()
            total_class = total_class + F.cross_entropy(
                pred_class_logits[batch_idx, pred_indices],
                gt_labels[gt_indices],
                reduction="mean",
            )
            matched_ious.append(matched_iou.mean())

        total_objectness = total_objectness + F.binary_cross_entropy_with_logits(
            pred_objectness[batch_idx],
            obj_targets,
            reduction="mean",
        )

    total_box = total_box / batch_size
    total_iou = total_iou / batch_size
    total_objectness = total_objectness / batch_size
    total_class = total_class / batch_size
    total_loss = (
        box_weight * total_box
        + iou_weight * total_iou
        + objectness_weight * total_objectness
        + class_weight * total_class
    )
    mean_iou = torch.stack(matched_ious).mean() if matched_ious else pred_boxes.new_tensor(0.0)
    return {
        "loss": total_loss,
        "box_loss": total_box,
        "iou_loss": total_iou,
        "objectness_loss": total_objectness,
        "class_loss": total_class,
        "matched_iou": mean_iou,
    }


def postprocess_detections(
    outputs: Dict[str, torch.Tensor],
    score_threshold: float = 0.5,
    nms_threshold: float = 0.5,
) -> List[Dict[str, torch.Tensor]]:
    batch_boxes = outputs["boxes"]
    batch_objectness = outputs["objectness_logits"].sigmoid()
    batch_class_probs = outputs["class_logits"].softmax(dim=-1)

    detections: List[Dict[str, torch.Tensor]] = []
    for boxes, objectness, class_probs in zip(batch_boxes, batch_objectness, batch_class_probs):
        class_scores, labels = class_probs.max(dim=-1)
        scores = class_scores * objectness
        boxes_xyxy = cxcywh_to_xyxy(boxes).clamp(0.0, 1.0)
        keep = scores >= score_threshold

        if keep.any():
            kept_boxes = boxes_xyxy[keep]
            kept_scores = scores[keep]
            kept_labels = labels[keep]
            kept_objectness = objectness[keep]
            keep_indices = nms(kept_boxes, kept_scores, nms_threshold)
            detections.append(
                {
                    "boxes": kept_boxes[keep_indices],
                    "scores": kept_scores[keep_indices],
                    "labels": kept_labels[keep_indices],
                    "objectness": kept_objectness[keep_indices],
                }
            )
        else:
            detections.append(
                {
                    "boxes": boxes_xyxy.new_zeros((0, 4)),
                    "scores": scores.new_zeros((0,)),
                    "labels": labels.new_zeros((0,), dtype=torch.long),
                    "objectness": objectness.new_zeros((0,)),
                }
            )
    return detections


def exact_detection_matches(
    pred_boxes: torch.Tensor,
    pred_labels: torch.Tensor,
    pred_scores: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    iou_threshold: float = 0.5,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
        return [], list(range(pred_boxes.shape[0])), list(range(gt_boxes.shape[0]))

    best_tp = -1
    best_score_sum = -1.0
    best_matches: List[Tuple[int, int]] = []
    pred_count = pred_boxes.shape[0]
    gt_count = gt_boxes.shape[0]
    max_pairs = min(pred_count, gt_count)
    iou_matrix = pairwise_iou(pred_boxes, gt_boxes)

    for match_count in range(max_pairs + 1):
        for pred_perm in itertools.permutations(range(pred_count), match_count):
            for gt_perm in itertools.permutations(range(gt_count), match_count):
                valid = True
                score_sum = 0.0
                for pred_idx, gt_idx in zip(pred_perm, gt_perm):
                    same_class = pred_labels[pred_idx].item() == gt_labels[gt_idx].item()
                    good_iou = iou_matrix[pred_idx, gt_idx].item() >= iou_threshold
                    if not (same_class and good_iou):
                        valid = False
                        break
                    score_sum += float(pred_scores[pred_idx].item())
                if valid:
                    if match_count > best_tp or (match_count == best_tp and score_sum > best_score_sum):
                        best_tp = match_count
                        best_score_sum = score_sum
                        best_matches = list(zip(pred_perm, gt_perm))

    matched_pred = {pred_idx for pred_idx, _ in best_matches}
    matched_gt = {gt_idx for _, gt_idx in best_matches}
    unmatched_pred = [idx for idx in range(pred_count) if idx not in matched_pred]
    unmatched_gt = [idx for idx in range(gt_count) if idx not in matched_gt]
    return best_matches, unmatched_pred, unmatched_gt


def update_average_precision_records(
    records: List[List[Tuple[float, int]]],
    gt_counts: Sequence[int],
    detections: Sequence[Dict[str, torch.Tensor]],
    gt_boxes_batch: torch.Tensor,
    gt_labels_batch: torch.Tensor,
    object_mask_batch: torch.Tensor,
    iou_threshold: float = 0.5,
) -> None:
    num_classes = len(records)
    for detection, gt_boxes, gt_labels, object_mask in zip(
        detections,
        gt_boxes_batch,
        gt_labels_batch,
        object_mask_batch,
    ):
        valid_gt_boxes = cxcywh_to_xyxy(gt_boxes[object_mask.bool()])
        valid_gt_labels = gt_labels[object_mask.bool()]
        for class_idx in range(num_classes):
            class_gt_boxes = valid_gt_boxes[valid_gt_labels == class_idx]
            gt_counts[class_idx] += int(class_gt_boxes.shape[0])

            class_pred_mask = detection["labels"] == class_idx
            class_pred_boxes = detection["boxes"][class_pred_mask]
            class_pred_scores = detection["scores"][class_pred_mask]
            if class_pred_boxes.numel() == 0:
                continue

            order = torch.argsort(class_pred_scores, descending=True)
            class_pred_boxes = class_pred_boxes[order]
            class_pred_scores = class_pred_scores[order]
            matched_gt = set()
            if class_gt_boxes.numel() == 0:
                for score in class_pred_scores:
                    records[class_idx].append((float(score.item()), 0))
                continue

            iou_matrix = pairwise_iou(class_pred_boxes, class_gt_boxes)
            for pred_idx, score in enumerate(class_pred_scores):
                best_iou = 0.0
                best_gt_idx = -1
                for gt_idx in range(class_gt_boxes.shape[0]):
                    if gt_idx in matched_gt:
                        continue
                    iou_value = float(iou_matrix[pred_idx, gt_idx].item())
                    if iou_value > best_iou:
                        best_iou = iou_value
                        best_gt_idx = gt_idx
                if best_iou >= iou_threshold and best_gt_idx >= 0:
                    matched_gt.add(best_gt_idx)
                    records[class_idx].append((float(score.item()), 1))
                else:
                    records[class_idx].append((float(score.item()), 0))


def compute_average_precision(records: Sequence[Tuple[float, int]], gt_count: int) -> float:
    if gt_count == 0:
        return 0.0
    if not records:
        return 0.0

    sorted_records = sorted(records, key=lambda item: item[0], reverse=True)
    scores = torch.tensor([item[0] for item in sorted_records], dtype=torch.float32)
    true_positives = torch.tensor([item[1] for item in sorted_records], dtype=torch.float32)
    del scores
    false_positives = 1.0 - true_positives

    cum_tp = torch.cumsum(true_positives, dim=0)
    cum_fp = torch.cumsum(false_positives, dim=0)
    precision = cum_tp / (cum_tp + cum_fp).clamp(min=1e-6)
    recall = cum_tp / max(float(gt_count), 1.0)

    precision = torch.cat((torch.tensor([1.0]), precision, torch.tensor([0.0])))
    recall = torch.cat((torch.tensor([0.0]), recall, torch.tensor([1.0])))
    for index in range(precision.shape[0] - 1, 0, -1):
        precision[index - 1] = torch.maximum(precision[index - 1], precision[index])

    recall_delta = recall[1:] - recall[:-1]
    return float((recall_delta * precision[1:]).sum().item())
