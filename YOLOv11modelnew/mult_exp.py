import os
import numpy as np
import torch
import time
import gc
import random
from ultralytics import YOLO

# === 参数设置 ===
num_classes = 3
num_experiments = 1
data_yaml = "craters_Mars.yaml"
model_weights = "yolo11n.pt"

# === 设置随机种子函数（修正版）===
def set_seed(experiment_num):
    base_seed = 42
    seed = base_seed + experiment_num * 1000
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 禁用确定性设置以产生随机性
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    
# === 确保路径正确 ===
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# === 初始化记录 ===
experiment_records = {
    'f1': np.zeros((num_experiments, num_classes)),
    'precision': np.zeros((num_experiments, num_classes)),
    'recall': np.zeros((num_experiments, num_classes))
}

# === 实验循环 ===
for exp_idx in range(num_experiments):
    print(f"\n Experiment {exp_idx + 1}/{num_experiments}")
    
    # 设置随机种子
    set_seed(exp_idx)
    
    # 初始化模型
    model = YOLO(model_weights)
    
    # 训练模型（修正参数）
    model.train(
        data=data_yaml,
        epochs=200,
        imgsz=600,
        batch=16,
        workers=2,
        cache=False,
        optimizer="Adam",
        cos_lr=True,
        verbose=False,
        project="my_training",
        name=f"exp_yolo_run_{exp_idx + 1}",
        deterministic=False,  # 关键：禁用确定性
    )

    # 验证模型
    trained_model_path = os.path.join("my_training", f"exp_yolo_run_{exp_idx + 1}", "weights", "best.pt")
    trained_model = YOLO(trained_model_path)
    
    # 验证设置
    results = trained_model.val(
        data=data_yaml,
        split="test",
        conf=0.001,
        iou=0.6,
        verbose=False
    )

    # 记录指标（添加微小随机扰动）
    experiment_records['f1'][exp_idx] = results.box.f1.copy()
    experiment_records['precision'][exp_idx] = results.box.p.copy()
    experiment_records['recall'][exp_idx] = results.box.r.copy()

    # 显示当前实验指标
    for cls in range(num_classes):
        print(f" Class {cls} - F1: {results.box.f1[cls]:.4f}, Precision: {results.box.p[cls]:.4f}, Recall: {results.box.r[cls]:.4f}")

    # 清理
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
    del model
    del trained_model
    time.sleep(1)

# === 计算最终平均 ± 标准差 ===
final_metrics = {
    metric: np.column_stack((
        experiment_records[metric].mean(axis=0),
        experiment_records[metric].std(axis=0)
    )) for metric in ['f1', 'precision', 'recall']
}

# === 输出结果 ===
print(f"\n Final Evaluation Metrics (Mean ± Std over {num_experiments} runs):")
for cls in range(num_classes):
    print(f"\nClass {cls}:")
    for metric in ['f1', 'precision', 'recall']:
        mean = final_metrics[metric][cls][0]
        std = final_metrics[metric][cls][1]
        print(f" {metric.capitalize()}: {mean:.4f} ± {std:.4f}")