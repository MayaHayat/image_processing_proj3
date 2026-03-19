import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from PIL import Image
import matplotlib.pyplot as plt
import os

# 1. Load Pretrained MobileNet V3 with the latest weights API
weights = MobileNet_V3_Large_Weights.DEFAULT
model = mobilenet_v3_large(weights=weights)
model.eval()

# 2. Use the automatic preprocessing transforms associated with these weights
preprocess = weights.transforms()
classes = weights.meta["categories"]

# 3. Your local image paths
images = ['goldfish.jpeg', 'jellyfish.jpeg', 'plane.jpeg']

plt.figure(figsize=(15, 5))

for i, img_path in enumerate(images):
    # Check if file exists to avoid errors
    if not os.path.exists(img_path):
        print(f"Warning: {img_path} not found. Please upload it to Colab.")
        continue

    # Load and preprocess
    img = Image.open(img_path).convert('RGB')
    batch = preprocess(img).unsqueeze(0) # Add batch dimension (1, 3, 224, 224)

    # Predict
    with torch.no_grad():
        prediction = model(batch).squeeze(0).softmax(0)
        class_id = prediction.argmax().item()
        score = prediction[class_id].item()
        category_name = classes[class_id]

    # Visualize
    plt.subplot(1, len(images), i+1)
    plt.imshow(img)
    plt.title(f"{category_name}\n({100*score:.1f}%)")
    plt.axis('off')

plt.tight_layout()
plt.show()

print("First 10 classes:")
print(weights.meta["categories"][100:200])