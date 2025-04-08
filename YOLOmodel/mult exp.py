import os
import numpy as np
import torch
import time
import gc
from ultralytics import YOLO

num_classes = 3
num_experiments = 30
data_yaml = "craters.yaml"
model_weights = "yolo11n.pt"

experiment_records = {
    'f1': np.zeros((num_experiments, num_classes)),
    'precision': np.zeros((num_experiments, num_classes)),
    'recall': np.zeros((num_experiments, num_classes))
}

for exp in range(num_experiments):
    print(f"\n🚀 Experiment {exp + 1}/{num_experiments}")

    model = YOLO(model_weights)

    model.train(
        data=data_yaml,
        epochs=200,
        imgsz=640,
        batch=16,
        workers=0,
        cache=True,
        optimizer="Adam",
        cos_lr=True,
        device=[0,],
        verbose=False,
        show=False,
        exist_ok=True
    )

    results = model.val(
        data=data_yaml,
        split="test",
        conf=0.001,
        iou=0.6,
        verbose=False
    )

    experiment_records['f1'][exp] = results.box.f1
    experiment_records['precision'][exp] = results.box.p
    experiment_records['recall'][exp] = results.box.r

    torch.cuda.empty_cache()
    gc.collect()
    del model
    time.sleep(3)

final_metrics = {
    metric: np.column_stack((
        experiment_records[metric].mean(axis=0),
        experiment_records[metric].std(axis=0)
    )) for metric in ['f1', 'precision', 'recall']
}

print("\n📊 Final Evaluation Metrics:")
for cls in range(num_classes):
    print(f"\nClass {cls}:")
    for metric in ['f1', 'precision', 'recall']:
        mean = final_metrics[metric][cls][0]
        std = final_metrics[metric][cls][1]
        print(f"  {metric.capitalize()}: {mean:.4f} ± {std:.4f}")


