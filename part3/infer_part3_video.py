from dataclasses import dataclass
import json
import os
from typing import List

import cv2
import torch
from PIL import Image

from part3_dataset import CLASS_NAMES, prepare_inference_image, project_boxes_to_original_image
from part3_losses import postprocess_detections
from part3_model import MobileNetV3FixedSlotDetector


BOX_COLORS = {
    "dog": (40, 110, 255),
    "person": (80, 200, 120),
}

@dataclass
class TrackState:
    track_id: int
    box: torch.Tensor
    label: int
    score: float
    missed_frames: int = 0
    pending_label: int | None = None
    pending_count: int = 0


def get_default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# Inference settings. Fill these in directly in the file.
CHECKPOINT_PATH = "./outputs/part3/run_20260317_190002/checkpoints/best.pt"
VIDEO_PATH = "doggirl.mp4"
OUTPUT_PATH = "doggirl_smooth.mp4"
SCORE_THRESHOLD = 0.45
NMS_THRESHOLD = 0.5
DEVICE = get_default_device()
MAX_FRAMES = None

# Tracking and smoothing defaults. Tune these here in the file.
BOX_SMOOTHING = 0.5
TRACK_IOU_THRESHOLD = 0.3
CLASS_SWITCH_PATIENCE = 3
MAX_MISSED_FRAMES = 2


def load_model(checkpoint_path: str, device: torch.device) -> tuple[MobileNetV3FixedSlotDetector, List[str], int]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_args = checkpoint.get("args", {})
    class_names = list(checkpoint.get("class_names", CLASS_NAMES))
    num_slots = int(checkpoint_args.get("max_objects", 3))
    dropout = float(checkpoint_args.get("dropout", 0.2))
    model = MobileNetV3FixedSlotDetector(
        num_classes=len(class_names),
        num_slots=num_slots,
        dropout=dropout,
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    image_size = int(checkpoint_args.get("image_size", 512))
    return model, class_names, image_size


def annotate_frame(
    frame_bgr,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    class_names: List[str],
) -> None:
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = [int(round(value)) for value in box.tolist()]
        class_name = class_names[int(label.item())]
        color = BOX_COLORS.get(class_name, (255, 255, 0))
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        text = f"{class_name} {float(score.item()):.2f}"
        cv2.putText(
            frame_bgr,
            text,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )


def compute_iou(box_a: torch.Tensor, box_b: torch.Tensor) -> float:
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    intersection = inter_w * inter_h
    if intersection <= 0.0:
        return 0.0

    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    union = area_a + area_b - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def smooth_box(previous_box: torch.Tensor, current_box: torch.Tensor, momentum: float) -> torch.Tensor:
    return previous_box * momentum + current_box * (1.0 - momentum)


def greedy_match_tracks(
    tracks: List[TrackState],
    detection_boxes: torch.Tensor,
    iou_threshold: float,
) -> List[tuple[int, int]]:
    candidate_matches: list[tuple[float, int, int]] = []
    for track_index, track in enumerate(tracks):
        for detection_index, detection_box in enumerate(detection_boxes):
            iou = compute_iou(track.box, detection_box)
            if iou >= iou_threshold:
                candidate_matches.append((iou, track_index, detection_index))

    candidate_matches.sort(reverse=True)
    matches: list[tuple[int, int]] = []
    used_track_indices: set[int] = set()
    used_detection_indices: set[int] = set()
    for _, track_index, detection_index in candidate_matches:
        if track_index in used_track_indices or detection_index in used_detection_indices:
            continue
        used_track_indices.add(track_index)
        used_detection_indices.add(detection_index)
        matches.append((track_index, detection_index))
    return matches


def apply_class_hysteresis(
    track: TrackState,
    detected_label: int,
    detected_score: float,
    switch_patience: int,
) -> None:
    if detected_label == track.label:
        track.score = detected_score
        track.pending_label = None
        track.pending_count = 0
        return

    if track.pending_label == detected_label:
        track.pending_count += 1
    else:
        track.pending_label = detected_label
        track.pending_count = 1

    if track.pending_count >= switch_patience:
        track.label = detected_label
        track.score = detected_score
        track.pending_label = None
        track.pending_count = 0
    else:
        # Keep the old class on short flips, but decay confidence slightly.
        track.score = max(track.score * 0.95, detected_score * 0.5)


def update_tracks(
    tracks: List[TrackState],
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    next_track_id: int,
    box_smoothing: float,
    track_iou_threshold: float,
    class_switch_patience: int,
    max_missed_frames: int,
) -> tuple[List[TrackState], int, torch.Tensor, torch.Tensor, torch.Tensor]:
    if boxes.numel() == 0:
        for track in tracks:
            track.missed_frames += 1
            track.pending_label = None
            track.pending_count = 0
            track.score *= 0.9
    else:
        matches = greedy_match_tracks(tracks, boxes, track_iou_threshold)
        matched_track_indices = {track_index for track_index, _ in matches}
        matched_detection_indices = {detection_index for _, detection_index in matches}

        for track_index, detection_index in matches:
            track = tracks[track_index]
            track.box = smooth_box(track.box, boxes[detection_index], box_smoothing)
            track.missed_frames = 0
            apply_class_hysteresis(
                track=track,
                detected_label=int(labels[detection_index].item()),
                detected_score=float(scores[detection_index].item()),
                switch_patience=class_switch_patience,
            )

        for track_index, track in enumerate(tracks):
            if track_index in matched_track_indices:
                continue
            track.missed_frames += 1
            track.pending_label = None
            track.pending_count = 0
            track.score *= 0.9

        for detection_index in range(boxes.shape[0]):
            if detection_index in matched_detection_indices:
                continue
            tracks.append(
                TrackState(
                    track_id=next_track_id,
                    box=boxes[detection_index].clone(),
                    label=int(labels[detection_index].item()),
                    score=float(scores[detection_index].item()),
                )
            )
            next_track_id += 1

    active_tracks = [track for track in tracks if track.missed_frames <= max_missed_frames]
    display_boxes = (
        torch.stack([track.box for track in active_tracks])
        if active_tracks
        else torch.empty((0, 4), dtype=torch.float32)
    )
    display_scores = torch.tensor([track.score for track in active_tracks], dtype=torch.float32)
    display_labels = torch.tensor([track.label for track in active_tracks], dtype=torch.long)
    return active_tracks, next_track_id, display_boxes, display_scores, display_labels


def main() -> None:
    device = torch.device(DEVICE)
    model, class_names, image_size = load_model(CHECKPOINT_PATH, device)

    if not CHECKPOINT_PATH or not VIDEO_PATH or not OUTPUT_PATH:
        raise ValueError("Set CHECKPOINT_PATH, VIDEO_PATH, and OUTPUT_PATH at the top of the file.")

    capture = cv2.VideoCapture(VIDEO_PATH)
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {VIDEO_PATH}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    writer = cv2.VideoWriter(
        OUTPUT_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    summary = {
        "video": VIDEO_PATH,
        "output": OUTPUT_PATH,
        "frames_processed": 0,
        "score_threshold": SCORE_THRESHOLD,
        "nms_threshold": NMS_THRESHOLD,
        "box_smoothing": BOX_SMOOTHING,
        "track_iou_threshold": TRACK_IOU_THRESHOLD,
        "class_switch_patience": CLASS_SWITCH_PATIENCE,
        "max_missed_frames": MAX_MISSED_FRAMES,
        "detections_per_frame": [],
    }

    frame_index = 0
    tracks: list[TrackState] = []
    next_track_id = 0
    while True:
        success, frame_bgr = capture.read()
        if not success:
            break
        if MAX_FRAMES is not None and frame_index >= MAX_FRAMES:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        image_tensor, meta = prepare_inference_image(pil_image, image_size=image_size)
        image_tensor = image_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            detections = postprocess_detections(
                outputs=outputs,
                score_threshold=SCORE_THRESHOLD,
                nms_threshold=NMS_THRESHOLD,
            )[0]

        canvas_boxes = detections["boxes"] * image_size
        original_boxes = project_boxes_to_original_image(canvas_boxes.cpu(), meta)
        tracks, next_track_id, display_boxes, display_scores, display_labels = update_tracks(
            tracks=tracks,
            boxes=original_boxes,
            scores=detections["scores"].cpu(),
            labels=detections["labels"].cpu(),
            next_track_id=next_track_id,
            box_smoothing=BOX_SMOOTHING,
            track_iou_threshold=TRACK_IOU_THRESHOLD,
            class_switch_patience=max(1, CLASS_SWITCH_PATIENCE),
            max_missed_frames=max(0, MAX_MISSED_FRAMES),
        )
        annotate_frame(
            frame_bgr=frame_bgr,
            boxes=display_boxes,
            scores=display_scores,
            labels=display_labels,
            class_names=class_names,
        )
        writer.write(frame_bgr)

        summary["detections_per_frame"].append(int(display_boxes.shape[0]))
        summary["frames_processed"] += 1
        frame_index += 1

    capture.release()
    writer.release()

    summary_path = os.path.splitext(OUTPUT_PATH)[0] + "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
