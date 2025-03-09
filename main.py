
from ultralytics import YOLO
from collections import Counter
import os
import cv2
import matplotlib.pyplot as plt






def load_class_names(label_folder):
    classes_file = os.path.join(label_folder, "classes.txt")
    if not os.path.exists(classes_file):
        print(f"{classes_file} file not exists!")
        return {}

    with open(classes_file, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f.readlines()]

    return {i: class_names[i] for i in range(len(class_names))}


def count_labels(label_folder):
    category_count = Counter()

    for label_file in os.listdir(label_folder):
        if label_file.endswith(".txt") and label_file != "classes.txt":
            with open(os.path.join(label_folder, label_file), "r", encoding="utf-8") as f:
                for line in f:
                    class_id = int(line.split()[0])
                    category_count[class_id] += 1

    return category_count


datasets = ["train", "test", "val"]
base_path = "final_datasets/label"

for dataset in datasets:
    label_folder = os.path.join(base_path, dataset)
    class_map = load_class_names(label_folder)
    counts = count_labels(label_folder)

    print(f"\n {dataset.upper()} dataset:")

    for class_id in sorted(class_map.keys()):
        class_name = class_map.get(class_id, f"class {class_id}")
        count = counts.get(class_id, 0)
        print(f"{class_name} (class {class_id}): {count}")


def visualize_annotations(image_path, label_path, class_map, class_colors):
    """
    Draw bounding boxes on images based on YOLO labels.
    """
    image = cv2.imread(image_path)
    h, w, _ = image.shape

    if not os.path.exists(label_path):
        print(f"⚠️ No label found for {image_path}")
        return

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center, y_center, box_w, box_h = map(float, parts[1:])

            x1 = int((x_center - box_w / 2) * w)
            y1 = int((y_center - box_h / 2) * h)
            x2 = int((x_center + box_w / 2) * w)
            y2 = int((y_center + box_h / 2) * h)

            label = class_map.get(class_id, f"Class {class_id}")
            color = class_colors.get(class_id, (255, 255, 255))  # Default to white if missing

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()


# Define dataset path
dataset_path = "final_datasets"
subset = "train"  # Change to 'test' or 'val' to visualize other sets

# Get paths
image_folder = os.path.join(dataset_path, "image", subset)
label_folder = os.path.join(dataset_path, "label", subset)

# Load class names
class_map = load_class_names(label_folder)

# Assign distinct colors to each class
class_colors = {
    0: (255, 0, 0),  # Large crater - Red
    1: (0, 255, 0),  # Small crater - Green
    2: (0, 0, 255),  # Medium crater - Blue
    3: (255, 255, 0)  # Incomplete crater - Yellow
}

# Get a list of image files (limit to first 3)
image_files = [f for f in os.listdir(image_folder) if f.endswith(".jpg")][:3]

if not image_files:
    print("⚠️ No images found in the dataset!")
else:
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        label_path = os.path.join(label_folder, image_file.replace(".jpg", ".txt"))
        visualize_annotations(image_path, label_path, class_map, class_colors)
