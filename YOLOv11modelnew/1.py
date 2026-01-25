import os
import numpy as np
import torch
import time
import gc
import random
from ultralytics import YOLO

num_classes = 3
num_experiments = 1
data_yaml = "craters_Mars.yaml"
model_weights = "yolo11n.pt"

# 验证模型
trained_model_path = os.path.join("my_training", f"exp_yolo_run_12", "weights", "best.pt")
trained_model = YOLO(trained_model_path)

# 验证设置
results = trained_model.val(
    data=data_yaml,
    split="test",
    conf=0.001,
    iou=0.6,
    verbose=False
)

# 显示当前实验指标
for cls in range(num_classes):
    print(f" Class {cls} - F1: {results.box.f1[cls]:.4f}, Precision: {results.box.p[cls]:.4f}, Recall: {results.box.r[cls]:.4f}")
