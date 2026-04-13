# crater-identification
Deep learning for crater identification

## Project Structure

```
CNNmodel/
├── CNNmodel.ipynb                # CNN uses the same dataset as ResNet50
├── Dataset/

Resnet50model/
├── Resnet50.ipynb                # ResNet50-based crater classifier
├── Dataset/

YOLOmodel/
├── YOLOv11.ipynb                 # YOLO-based crater detector
├── craters.yaml                  # YOLO dataset config
├── yolo11n.pt                    # Trained YOLOv11 weights
├── Dataset/                     # YOLO dataset folder (images + labels)
├── runs/detect/                  # Inference output images
│   ├── Predicted_1.jpg
│   └── Predicted_2.jpg
├── resized_4k.jpg
├── mult_exp.py

README.md                         # Project documentation
```

## Dataset Setup

Original data: 
*   Mars Reconnaissance Orbiter HiRISE data, NASA Planetary Data System: [link](https://pds-imaging.jpl.nasa.gov/search/?fq=MRO_IMAGE_CLASS%3Acrater&amp;fq=CRATER_COUNTS%3A%5B1%20TO%20131%5D&amp;fq=-ATLAS_THUMBNAIL_URL%3Abrwsnotavail.jpg&amp;q=crater)

You must manually place your dataset inside each model folder as follows:

### Folder structure for `Resnet50model/`, `CNNmodel/`, and `YOLOmodel/`:

```
Dataset/
├── Mars
|   ├── images
│       ├── train/
│       ├── val/
│       └── test/
|   ├── labels/
│       ├── train/
│       ├── val/
│       └── test/
├── Moon
|   ├── images
│       ├── train/
│       ├── val/
│       └── test/
|   ├── labels/
│       ├── train/
│       ├── val/
│       └── test/
```

Each image should have a corresponding YOLO-format label file in the `labels/` folder.  
Make sure all three folders (`Resnet50model/Dataset/`, `CNNmodel/Dataset/`, and `YOLOmodel/Dataset/`) follow this structure.

> CNN and ResNet50 share the same dataset structure, so you can copy the same folder into both.

## Models Overview

- **CNNmodel.ipynb**  
  A simple CNN architecture to classify crater size categories.

- **Resnet50.ipynb**  
  A crater classifier based on the ResNet50 architecture, trained from scratch without using ImageNet weights. The top layers were customized for four-class classification.

- **YOLOv11.ipynb**  
  Object detection model (YOLOv11) for locating and classifying craters.

## Dependencies

Install dependencies with:

pip install -r requirements.txt

Or manually install key packages:

pip install numpy opencv-python tensorflow keras matplotlib yolov5

## Getting Started

Each notebook is standalone. To run:
1. Open the desired `.ipynb` file.
2. Ensure the dataset is in the correct subfolder.
3. Run all cells.

## Example Outputs

YOLO predictions can be found under `YOLOv11model/runs/detect/`.

## Authors

- Yihan Ma (UNSW)
- Jinghong Liang (UNSW)
- Jessie Guo (UNSW)
