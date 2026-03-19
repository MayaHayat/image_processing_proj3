import torch

def calculate_iou(preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    x1 = torch.max(preds[:, 0], labels[:, 0])
    y1 = torch.max(preds[:, 1], labels[:, 1])
    x2 = torch.min(preds[:, 2], labels[:, 2])
    y2 = torch.min(preds[:, 3], labels[:, 3])
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    area_pred = (preds[:, 2] - preds[:, 0]).clamp(0) * (preds[:, 3] - preds[:, 1]).clamp(0)
    area_gt = (labels[:, 2] - labels[:, 0]).clamp(0) * (labels[:, 3] - labels[:, 1]).clamp(0)
    union = area_pred + area_gt - intersection + 1e-6
    return (intersection / union).mean()
