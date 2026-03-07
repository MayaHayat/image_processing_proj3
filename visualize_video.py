import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import cv2

# --- Helpers (match training script) ---
def get_device():
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class MobileNetDetector(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.mobilenet_v3_large(weights='DEFAULT')
        self.features = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.regressor = nn.Sequential(
            nn.Linear(960, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.regressor(x)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def main():
    # --- 1. Setup Paths and Model ---
    VIDEO_PATH = './eevee3.mp4'
    OUTPUT_PATH = './detected_dogs.mp4'
    MODEL_PATH = 'dogs/LR_1e-3_BS_32/best_model.pth'

    device = get_device()
    model = MobileNetDetector().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # --- 2. Video Initialize ---
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {VIDEO_PATH}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    limit_seconds = 60
    max_frames = int(fps * limit_seconds)

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    print(f"Processing first {limit_seconds} seconds ({max_frames} frames)...")

    # --- 3. Loop with Frame Counter ---
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_count >= max_frames:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        img_tensor = transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(img_tensor).squeeze().cpu().numpy()

        x1 = int(pred[0] * width)
        y1 = int(pred[1] * height)
        x2 = int(pred[2] * width)
        y2 = int(pred[3] * height)
        # Clamp to frame bounds
        x1, x2 = max(0, min(x1, width)), max(0, min(x2, width))
        y1, y2 = max(0, min(y1, height)), max(0, min(y2, height))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        out.write(frame)
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Progress: {frame_count}/{max_frames} frames")

    cap.release()
    out.release()
    print(f"Done! 1-minute clip saved to: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()

