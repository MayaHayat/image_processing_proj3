"""
Run the current part2 detector on a video and save an annotated copy
with predicted boxes.
"""
import argparse
import os
from typing import Dict, Optional, Tuple

import cv2
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from part2_new import MobileNetDetector


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VIDEO_PATH = os.path.join(BASE_DIR, "eevee3.mp4")
DEFAULT_OUTPUT_ROOT = os.path.join(BASE_DIR, "outputs", "part2")
DEFAULT_OUTPUT_PATH = os.path.join(BASE_DIR, "eevee3_detected3.mp4")
DEFAULT_IMAGE_SIZE = 224
DEFAULT_SCORE_THRESHOLD = 0.45
DEFAULT_SMOOTH_ALPHA = 0.7
DEFAULT_SMOOTH_MAX_GAP = 3
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_LABEL = "dog"


def get_device() -> torch.device:
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize part2 video detections.")
    parser.add_argument("--video", default=DEFAULT_VIDEO_PATH, help="Input video path.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Output annotated video path.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path. Defaults to the newest model checkpoint in outputs/part2.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        help="Override the detection score threshold stored in the checkpoint.",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=300.0,
        help="Maximum number of seconds to process.",
    )
    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=DEFAULT_SMOOTH_ALPHA,
        help="EMA factor for temporal box smoothing. Higher means smoother but more lag.",
    )
    parser.add_argument(
        "--smooth-max-gap",
        type=int,
        default=DEFAULT_SMOOTH_MAX_GAP,
        help="Keep drawing the last smoothed box for this many missed frames.",
    )
    return parser.parse_args()


def find_latest_checkpoint(root_dir: str) -> Optional[str]:
    if not os.path.isdir(root_dir):
        return None

    candidates = []
    for dirpath, _, filenames in os.walk(root_dir):
        for priority, file_name in enumerate(("best.pt", "best_iou.pt", "last.pt", "best_model.pth", "best_model_iou.pth", "model.pth")):
            if file_name in filenames:
                path = os.path.join(dirpath, file_name)
                candidates.append((os.path.getmtime(path), -priority, path))

    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][2]


def load_checkpoint(path: str, device: torch.device) -> Tuple[Dict[str, torch.Tensor], Dict[str, object]]:
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        return checkpoint["model_state"], checkpoint
    if isinstance(checkpoint, dict):
        return checkpoint, {}
    raise ValueError(f"Unsupported checkpoint format: {path}")


def preprocess_frame(frame_bgr, image_size: int) -> torch.Tensor:
    pil_image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    resized_image = TF.resize(pil_image, [image_size, image_size])
    image_tensor = TF.to_tensor(resized_image)
    image_tensor = TF.normalize(image_tensor, IMAGENET_MEAN, IMAGENET_STD)
    return image_tensor.unsqueeze(0)


def map_box_to_original_frame(
    box_xyxy_norm: torch.Tensor,
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    box_xyxy_norm = box_xyxy_norm.clamp(0.0, 1.0)
    x1 = max(0, min(int(round(float(box_xyxy_norm[0].item()) * width)), width))
    y1 = max(0, min(int(round(float(box_xyxy_norm[1].item()) * height)), height))
    x2 = max(0, min(int(round(float(box_xyxy_norm[2].item()) * width)), width))
    y2 = max(0, min(int(round(float(box_xyxy_norm[3].item()) * height)), height))
    return x1, y1, x2, y2


def extract_boxes_and_scores(
    outputs,
    score_threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(outputs, dict):
        boxes = outputs["boxes"].detach().cpu().reshape(-1, 4).clamp(0.0, 1.0)
        if "objectness_logits" in outputs:
            scores = outputs["objectness_logits"].detach().cpu().reshape(-1).sigmoid()
        else:
            scores = torch.ones((boxes.shape[0],), dtype=torch.float32)
    else:
        boxes = outputs.detach().cpu().reshape(-1, 4).clamp(0.0, 1.0)
        scores = torch.ones((boxes.shape[0],), dtype=torch.float32)

    keep = scores >= score_threshold
    return boxes[keep], scores[keep]


class TemporalDetectionSmoother:
    def __init__(self, alpha: float, max_gap: int) -> None:
        self.alpha = float(max(0.0, min(alpha, 0.999)))
        self.max_gap = max(0, int(max_gap))
        self.smoothed_box: Optional[torch.Tensor] = None
        self.smoothed_score: Optional[float] = None
        self.missed_frames = 0

    def update(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[float]]:
        if boxes.numel() == 0 or scores.numel() == 0:
            self.missed_frames += 1
            if self.smoothed_box is not None and self.missed_frames <= self.max_gap:
                return self.smoothed_box.clone(), self.smoothed_score
            self.smoothed_box = None
            self.smoothed_score = None
            return None, None

        best_index = int(torch.argmax(scores).item())
        current_box = boxes[best_index].clone().float()
        current_score = float(scores[best_index].item())

        if self.smoothed_box is None:
            self.smoothed_box = current_box
            self.smoothed_score = current_score
        else:
            self.smoothed_box = self.alpha * self.smoothed_box + (1.0 - self.alpha) * current_box
            previous_score = self.smoothed_score if self.smoothed_score is not None else current_score
            self.smoothed_score = self.alpha * previous_score + (1.0 - self.alpha) * current_score

        self.missed_frames = 0
        return self.smoothed_box.clone(), self.smoothed_score


def main() -> None:
    args = parse_args()
    device = get_device()
    checkpoint_path = args.checkpoint or find_latest_checkpoint(DEFAULT_OUTPUT_ROOT)

    if checkpoint_path is None:
        raise FileNotFoundError(
            "No checkpoint found under part2 outputs. "
            "Pass --checkpoint path/to/checkpoints/best.pt after training part2_new."
        )
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Video not found: {args.video}")

    state_dict, checkpoint_meta = load_checkpoint(checkpoint_path, device)
    checkpoint_args = checkpoint_meta.get("args", {}) if isinstance(checkpoint_meta, dict) else {}
    image_size = int(checkpoint_args.get("image_size", DEFAULT_IMAGE_SIZE))
    score_threshold = (
        float(args.score_threshold)
        if args.score_threshold is not None
        else float(checkpoint_args.get("score_threshold", DEFAULT_SCORE_THRESHOLD))
    )

    model = MobileNetDetector().to(device)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Loaded checkpoint: {checkpoint_path}")
    print(f"[INFO] Image size: {image_size}, score threshold: {score_threshold:.2f}")
    print(
        f"[INFO] Smoothing alpha: {max(0.0, min(args.smooth_alpha, 0.999)):.2f}, "
        f"max gap: {max(0, args.smooth_max_gap)}"
    )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        fps = 30.0
    max_frames = int(fps * max(args.max_seconds, 0.0))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create output video: {args.output}")

    print(
        f"[INFO] Processing {min(max_frames, total_frames) if total_frames > 0 else max_frames} "
        f"frames -> {args.output}"
    )

    frame_count = 0
    label = str(checkpoint_args.get("label_name", DEFAULT_LABEL))
    box_color = (0, 255, 0)
    smoother = TemporalDetectionSmoother(
        alpha=args.smooth_alpha,
        max_gap=args.smooth_max_gap,
    )

    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok or frame_count >= max_frames:
                break

            image_tensor = preprocess_frame(frame, image_size)
            image_tensor = image_tensor.to(device)

            with torch.no_grad():
                outputs = model(image_tensor)
            pred_boxes, pred_scores = extract_boxes_and_scores(outputs, score_threshold)
            smoothed_box, smoothed_score = smoother.update(pred_boxes, pred_scores)

            if smoothed_box is not None and smoothed_score is not None:
                x1, y1, x2, y2 = map_box_to_original_frame(smoothed_box, width, height)
                if x2 - x1 >= 5 and y2 - y1 >= 5:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
                    cv2.putText(
                        frame,
                        f"{label} {smoothed_score:.2f}",
                        (x1, max(y1 - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        box_color,
                        2,
                    )

            writer.write(frame)
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"  {frame_count}/{min(max_frames, total_frames) if total_frames > 0 else max_frames} frames")
    finally:
        cap.release()
        writer.release()

    print(f"[DONE] Saved annotated video to: {args.output}")


if __name__ == "__main__":
    main()
