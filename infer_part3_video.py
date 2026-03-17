import argparse
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


def get_default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Part 3 detector inference on a video.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--score-threshold", type=float, default=0.45)
    parser.add_argument("--nms-threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default=get_default_device())
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model, class_names, image_size = load_model(args.checkpoint, device)

    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    summary = {
        "video": args.video,
        "output": args.output,
        "frames_processed": 0,
        "score_threshold": args.score_threshold,
        "nms_threshold": args.nms_threshold,
        "detections_per_frame": [],
    }

    frame_index = 0
    while True:
        success, frame_bgr = capture.read()
        if not success:
            break
        if args.max_frames is not None and frame_index >= args.max_frames:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        image_tensor, meta = prepare_inference_image(pil_image, image_size=image_size)
        image_tensor = image_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            detections = postprocess_detections(
                outputs=outputs,
                score_threshold=args.score_threshold,
                nms_threshold=args.nms_threshold,
            )[0]

        canvas_boxes = detections["boxes"] * image_size
        original_boxes = project_boxes_to_original_image(canvas_boxes.cpu(), meta)
        annotate_frame(
            frame_bgr=frame_bgr,
            boxes=original_boxes,
            scores=detections["scores"].cpu(),
            labels=detections["labels"].cpu(),
            class_names=class_names,
        )
        writer.write(frame_bgr)

        summary["detections_per_frame"].append(int(original_boxes.shape[0]))
        summary["frames_processed"] += 1
        frame_index += 1

    capture.release()
    writer.release()

    summary_path = os.path.splitext(args.output)[0] + "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
