import urllib.request
import zipfile
import os
import os
import json
import requests
from pycocotools.coco import COCO

# --- Configuration ---
ANNOTATION_FILE = 'annotations/instances_train2017.json'  # Path to the downloaded COCO annotations
OUTPUT_DIR = 'custom_coco_dataset'            # Folder to save images
OUTPUT_JSON = 'custom_annotations.json'       # File to save filtered annotations
CLASSES_TO_KEEP = ['person', 'dog']         # The classes you want

def main():
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading COCO annotations (this might take a minute)...")
    coco = COCO(ANNOTATION_FILE)

    # 1. Get Category IDs for 'person' and 'dog'
    catIds = coco.getCatIds(catNms=CLASSES_TO_KEEP)
    print(f"Category IDs found: {catIds}")

    # 2. Get all Image IDs that contain AT LEAST ONE of these categories
    imgIds = set()
    for catId in catIds:
        imgIds.update(coco.getImgIds(catIds=[catId]))
    imgIds = list(imgIds)
    print(f"Found {len(imgIds)} images containing people or dogs.")

    # 3. Filter to images that contain ONLY people or dogs (no other classes)
    filtered_imgIds = []
    for imgId in imgIds:
        annIds = coco.getAnnIds(imgIds=[imgId])
        anns = coco.loadAnns(annIds)
        categories = set(ann['category_id'] for ann in anns)
        if categories.issubset(set(catIds)) and categories:  # Only person/dog, and at least one
            if len(anns) < 4:  # Fewer than 4 objects total
                filtered_imgIds.append(imgId)
    imgIds = filtered_imgIds
    print(f"After filtering, {len(imgIds)} images contain only people or dogs with fewer than 4 objects.")
    
    # 3. Download the images
    images_metadata = coco.loadImgs(imgIds)
    downloaded_images = []
    
    print("Downloading images...")
    for i, img in enumerate(images_metadata):
        img_url = img['coco_url']
        file_name = img['file_name']
        file_path = os.path.join(OUTPUT_DIR, file_name)

        # Download if it doesn't already exist
        if not os.path.exists(file_path):
            try:
                img_data = requests.get(img_url).content
                with open(file_path, 'wb') as handler:
                    handler.write(img_data)
                downloaded_images.append(img)
            except Exception as e:
                print(f"Error downloading {file_name}: {e}")
        else:
            downloaded_images.append(img)
            
        if (i + 1) % 500 == 0:
            print(f"Downloaded {i + 1}/{len(images_metadata)} images.")

    # 4. Filter annotations to only include our selected images and categories
    print("Filtering annotations...")
    annIds = coco.getAnnIds(imgIds=imgIds, catIds=catIds, iscrowd=None)
    annotations = coco.loadAnns(annIds)

    # 5. Build and save the new, smaller JSON dataset
    custom_dataset = {
        "info": coco.dataset.get('info', {}),
        "licenses": coco.dataset.get('licenses', []),
        "images": downloaded_images,
        "annotations": annotations,
        "categories": coco.loadCats(catIds)
    }

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(custom_dataset, f)
        
    print(f"Done! Saved {len(downloaded_images)} images to '{OUTPUT_DIR}' and annotations to '{OUTPUT_JSON}'.")

if __name__ == '__main__':
    url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    zip_path = "annotations_trainval2017.zip"
    
    print("Downloading annotations (this is ~241MB and might take a few minutes)...")
    # This bypasses browser security blocks by downloading directly
    urllib.request.urlretrieve(url, zip_path)
    
    print("Download complete! Extracting the files...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(".") # Extracts into your current folder
    
    # Clean up the zip file to save space
    os.remove(zip_path)
    
    print("Done! You should now see a folder named 'annotations' in your directory.")
    main()